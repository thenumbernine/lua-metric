#!/usr/bin/env luajit
require 'ext'
local ImGuiApp = require 'imguiapp'
local bit = require 'bit'
local ffi = require 'ffi'
local ig = require 'ffi.imgui'
local gl = require 'ffi.OpenGL'
local sdl = require 'ffi.sdl'
local GLTex2D = require 'gl.tex2d'
local Image = require 'image'
local Mouse = require 'gui.mouse'
local vec3 = require 'vec.vec3'
local quat = require 'vec.quat'
local symmath = require 'symmath'
symmath.tostring = require 'symmath.tostring.SingleLine'

local App = class(ImGuiApp)

local eqnPtr = ffi.new('int[1]', 0)

local params = table{
	{var=symmath.var'u', divs=300, min=-5, max=5},
	{var=symmath.var'v', divs=300, min=-5, max=5},
}
local u, v = params[1].var, params[2].var
local eqns = table{
	{name='x', expr=u},
	{name='y', expr=v},
	{name='z', expr=0},
}

local options = table{
	{
		name = 'Cartesian',
		consts = {},
		vars = {'u', 'v'},
		mins = {-2, -2},
		maxs = {2, 2},
		exprs = {'u', 'v', '0'},
	},
	{
		name = 'Spherical',
		consts = {r=1, pi=math.pi},
		mins = {0, -12},
		maxs = {12, 12},
		vars = {'theta', 'phi'},
		exprs = {
			'r * sin(theta * pi / 12) * cos(phi * pi / 12)',
			'r * sin(theta * pi / 12) * sin(phi * pi / 12)',
			'r * cos(theta * pi / 12)',
		},
	},
	{
		name = 'Polar',
		consts = {pi=math.pi},
		mins = {0, -12},
		maxs = {8, 12},
		vars = {'r', 'theta'},
		exprs = {
			'r / 4 * cos(theta * pi / 4)',
			'r / 4 * sin(theta * pi / 4)',
			'0',
		},
	},
	{
		name = 'Torus',
		consts = {r=.25, R=1, pi=math.pi},
		mins = {-4, -12},
		maxs = {4, 12},
		vars = {'theta', 'phi'},
		exprs = {
			'(r * sin(theta * pi / 4) + R) * cos(phi * pi / 12)',
			'(r * sin(theta * pi / 4) + R) * sin(phi * pi / 12)',
			'r * cos(theta * pi / 4)',
		},
	},
	{
		name = 'Paraboloid',
		consts = {},
		mins = {-1, -1},
		maxs = {1, 1},
		vars = {'u', 'v'},
		exprs = {'u', 'v', '-u^2 -v^2'},
	},
	{
		name = 'Hyperboloid',
		consts = {},
		mins = {-1, -1},
		maxs = {1, 1},
		vars = {'u', 'v'},
		exprs = {'u', 'v', 'u^2 - v^2'},
	},
}

local controls = table{'rotate', 'select', 'direct'}
local controlIndexes = controls:map(function(v,k) return k,v end)

function App:init(...)
	App.super.init(self, ...)

	self.consts = {}
end

function App:initGL(...)
	App.super.initGL(self, ...)

	gl.glEnable(gl.GL_DEPTH_TEST)

--[[ tex: todo just render the shader to the tex and you'll get mipmapping!
	local width, height = 64, 64
	self.tex = GLTex2D{
		image = Image(width, height, 4, 'unsigned char', function(i,j)
			if i == 0 
			or i == width-1
			or j == 0 
			or j == height-1
			then
				return 0,0,0,255
			else
				return 0,127,127,255
			end
		end),
		minFilter = gl.GL_LINEAR_MIPMAP_LINEAR,	
		magFilter = gl.GL_LINEAR,	
		generateMipmap = true,
	}
--]]	
	local GLProgram = require 'gl.program'
	self.shader = GLProgram{
		vertexCode = [[
varying vec2 intCoordV;
varying vec3 normalV;
varying vec3 vertexV;
void main() {
	normalV = (gl_ModelViewMatrix * vec4(gl_Normal, 0.)).xyz;
	vec4 mvtx = gl_ModelViewMatrix * gl_Vertex;
	vertexV = mvtx.xyz;
	gl_Position = gl_ProjectionMatrix * mvtx;
	intCoordV = gl_MultiTexCoord0.st;
}
]],
		fragmentCode = [[
varying vec2 intCoordV;
varying vec3 normalV;
varying vec3 vertexV;
void main() {
	vec3 n = normalize(normalV);
	if (n.z < 0.) n = -n;	//backface lighting
	
	vec2 fc = mod(intCoordV.xy, 1.); //grid
	float i = 1. - 8. * fc.x * fc.y * (1. - fc.x) * (1. - fc.y);
	i = pow(i, 50.);
	
	gl_FragColor = vec4(.25, .5, .5, 1.);
	gl_FragColor.rgb *= 1. - i;
	vec3 u = normalize(vertexV);
	float l = dot(n, u);
	gl_FragColor.rgb *= max(abs(l), .3);
}
]],
	}

	self:setOption(options[1])
	
	self.controlPtr = ffi.new('int[1]', controlIndexes.rotate-1)
end

function App:calculateMesh()
	
	if self.displayList then
		gl.glDeleteLists(self.displayList, 1)
	end

	-- refresh ...

	local vars = params:map(function(param) return param.var end)
	for _,eqn in ipairs(eqns) do
		local expr = symmath.clone(eqn.expr)()
		for var,value in pairs(self.consts) do
			expr = expr:replace(var, value)()
		end
		eqn.func = expr:compile(vars)
	end

	do
		local x,y,z = symmath.vars('x','y','z')
		local flatCoords = {x,y,z}
		local curvedCoords = vars
		local Tensor = symmath.Tensor
		Tensor.coords{
			{variables=flatCoords, symbols='IJKLMN'},
			{variables=curvedCoords},
		}
		local eta = Tensor('_IJ', {1,0,0},{0,1,0},{0,0,1})
		Tensor.metric(eta, eta, 'I')

		local u = Tensor('^I', eqns:map(function(eqn) return eqn.expr end):unpack())
		local e = Tensor'_u^I'
		e['_u^I'] = u'^I_,u'()
		local g = (e'_u^I' * e'_v^J' * eta'_IJ')()
		Tensor.metric(g)
		local dg = g'_uv,w'()
		local Gamma = ((dg'_uvw' + dg'_uwv' - dg'_vwu')/2)()
		Gamma = Gamma'^u_vw'()
		self.strs = table()
		for i,xi in ipairs(vars) do
			for j,xj in ipairs(vars) do
				if g[i][j] ~= symmath.Constant(0) then
					self.strs:insert('g_'..xi..'_'..xj..' = '..g[i][j])
				end
			end
		end
		for i,xi in ipairs(vars) do
			for j,xj in ipairs(vars) do
				for k,xk in ipairs(vars) do
					if Gamma[i][j][k] ~= symmath.Constant(0) then
						self.strs:insert('Gamma^'..xi..'_'..xj..'_'..xk..' = '..Gamma[i][j][k])
					end
				end
			end
		end
		for _,str in ipairs(self.strs) do
			print(str)
		end
	end

	self.displayList = gl.glGenLists(1)
	gl.glNewList(self.displayList, gl.GL_COMPILE)

	local function getpt(uvalue, vvalue)
		local pt = vec3()
		for i=1,3 do
			if eqns[i] then 
				pt[i] = eqns[i].func(uvalue, vvalue) 
			end
		end
		return pt
	end

	self.mesh = table()

	local u, v = params:unpack()
	local du = (u.max - u.min) / u.divs
	local dv = (v.max - v.min) / v.divs
	for udiv=0,u.divs-1 do
		gl.glBegin(gl.GL_TRIANGLE_STRIP)
		for vdiv=0,v.divs do
			local vfrac = vdiv / v.divs
			local vvalue = vfrac * (v.max - v.min) + v.min
			for uofs=1,0,-1 do
				local ufrac = (udiv + uofs) / u.divs
				local uvalue = ufrac * (u.max - u.min) + u.min
				local pt = getpt(uvalue, vvalue)
				gl.glTexCoord2f(uvalue, vvalue)
				
				local dp_du = ((getpt(uvalue + du, vvalue) - getpt(uvalue - du, vvalue))):normalize()
				local dp_dv = ((getpt(uvalue, vvalue + dv) - getpt(uvalue, vvalue - dv))):normalize()
				local n = vec3.cross(dp_du, dp_dv):normalize()
				gl.glNormal3f(n:unpack())
			
				self.mesh:insert{
					p=pt,
					dp_du=dp_du,
					dp_dv=dp_dv,
					n=n,
				}
				gl.glVertex3f(pt:unpack())
			end
		end
		gl.glEnd()
	end
	
	gl.glEndList()
end

local mouse = Mouse()
local leftShiftDown
local rightShiftDown 
local viewDist = 3
local zoomFactor = .1
local zNear = .1
local zFar = 100
local tanFovX = 1
local tanFovY = 1
local viewScale = 1
local viewPos = vec3()
local viewAngle = quat()

function App:event(event)
	if ig.igGetIO()[0].WantCaptureKeyboard then return end
	if event.type == sdl.SDL_MOUSEBUTTONDOWN then
		if event.button.button == sdl.SDL_BUTTON_WHEELUP then
			viewDist = viewDist * zoomFactor
		elseif event.button.button == sdl.SDL_BUTTON_WHEELDOWN then
			viewDist = viewDist / zoomFactor
		end
	elseif event.type == sdl.SDL_KEYDOWN or event.type == sdl.SDL_KEYUP then
		if event.key.keysym.sym == sdl.SDLK_LSHIFT then
			leftShiftDown = event.type == sdl.SDL_KEYDOWN
		elseif event.key.keysym.sym == sdl.SDLK_RSHIFT then
			rightShiftDown = event.type == sdl.SDL_KEYDOWN
		end
	end
end

function App:setOption(option)
	self.consts = table.map(option.consts, function(v,k)
		return v, symmath.var(k)
	end)
	for i=1,#option.vars do
		params[i].min = option.mins[i]
		params[i].max = option.maxs[i]
	end
	local vars = params:map(function(param) return param.var end)
	local varnames = table(option.vars)
	for varname,value in pairs(option.consts) do
		varnames:insert(varname)
		vars:insert(symmath.var(varname))
	end
	for i,eqn in ipairs(eqns) do
		local expr = option.exprs[i]
		print('from',expr)
		local str = table{
			'local '..varnames:concat','..' = ...',
			"local symmath = require 'symmath'",
			'local sin = symmath.sin',
			'local cos = symmath.cos',
			'return '..expr,
		}:concat'\n'
		print('str',str)
		eqn.expr = assert(loadstring(str))(vars:unpack())
		print('to',eqn.expr)
	end
	self:calculateMesh()
end

function App:updateGUI()
	if ig.igCollapsingHeader'controls:' then
		for i,control in ipairs(controls) do	
			ig.igRadioButton(control, self.controlPtr, i-1)
		end
	end
	if ig.igCollapsingHeader'predefined:' then
		for _,option in ipairs(options) do
			if ig.igButton(option.name) then
				self:setOption(option)
			end
		end
	end
	if ig.igCollapsingHeader'parameters:' then
		local remove
		for i,param in ipairs(params) do
			ig.igPushIdStr('parameter '..i)
			--[[ 2D only for now
			if ig.igButton'-' then
				remove = remove or table()
				remove:insert(i)
			end
			--]]
			ig.igSameLine()
			ig.igText(param.var.name)
			local int = ffi.new('int[1]', param.divs)
			if ig.igInputInt('divs', int) then
				param.divs = int[0]
			end
			local float = ffi.new('float[1]', param.min)
			if ig.igInputFloat('min', float) then
				param.min = float[0]
			end
			float[0] = param.max
			if ig.igInputFloat('max', float) then
				param.max = float[0]
			end
			ig.igPopId()
		end
		if remove then
			for i=#remove,1,-1 do
				params:remove(remove[i])
			end
		end
	end
	if ig.igCollapsingHeader'equations:' then
		local remove
		for i,eqn in ipairs(eqns) do
			ig.igPushIdStr('equation '..i)
			if ig.igButton'-' then
				remove = remove or table()
				remove:insert(i)
			end
			ig.igSameLine()
			ig.igText(eqn.name..' = '..eqn.expr)
			ig.igPopId()
		end
		if remove then
			for i=#remove,1,-1 do
				eqns:remove(remove[i])
			end
		end
	end
	if ig.igCollapsingHeader'derived variables:' then
		for _,str in ipairs(self.strs) do
			ig.igText(str)
		end
	end
end

function App:update()
	local w, h = self:size()
	local ar = w / h
	
	gl.glClear(bit.bor(gl.GL_COLOR_BUFFER_BIT, gl.GL_DEPTH_BUFFER_BIT))
	
	mouse:update()

	local ray	-- = mouse point and direction
	local function lineRayDist(v, ray)
		local src = viewPos 
		local dir = viewAngle:rotate(vec3(ar * (mouse.pos[1]*2-1), mouse.pos[2]*2-1, -1))
		local t = math.max(0, (v - src):dot(dir) / dir:lenSq())
		return (src + t * dir - v):length()
	end
	if not ig.igGetIO()[0].WantCaptureKeyboard then 
		if self.controlPtr[0] == controlIndexes.rotate-1 then
			if mouse.leftDragging then
				if leftShiftDown or rightShiftDown then
					viewDist = viewDist * math.exp(100 * zoomFactor * mouse.deltaPos[2])
				else
					local magn = mouse.deltaPos:length() * 1000
					if magn > 0 then
						local normDelta = mouse.deltaPos / magn
						local r = quat():fromAngleAxis(-normDelta[2], normDelta[1], 0, -magn)
						viewAngle = (viewAngle * r):normalize()
					end
				end
			end
		elseif self.controlPtr[0] == controlIndexes.select-1 then
			if mouse.leftClick then
				-- find closest point on mesh via mouseray
				self.selectedVtx = self.mesh:map(function(v)
					return lineRayDist(v.pt, ray)
				end):inf()
			end	
		elseif self.controlPtr[0] == controlIndexes.direct-1 then
			if mouse.leftClick then
				-- find closest point on mesh via mouseray
				local _, bestPt = self.mesh:map(function(v)
					return lineRayDist(v.pt, ray)
				end):inf()
				self.dir = bestPt - self.selectedVtx
					-- ... reprojected onto the surface
			end	
		end
	end
		
	viewPos = viewAngle:zAxis() * viewDist 
	
	gl.glClear(gl.GL_DEPTH_BUFFER_BIT + gl.GL_COLOR_BUFFER_BIT)
	gl.glMatrixMode(gl.GL_PROJECTION)
	gl.glLoadIdentity()
	gl.glFrustum(-zNear * ar * tanFovX, zNear * ar * tanFovX, -zNear * tanFovY, zNear * tanFovY, zNear, zFar);

	gl.glMatrixMode(gl.GL_MODELVIEW)
	gl.glLoadIdentity()
	gl.glScaled(viewScale, viewScale, viewScale)
	local aa = viewAngle:toAngleAxis()
	gl.glRotated(-aa[4], aa[1], aa[2], aa[3])
	gl.glTranslated(-viewPos[1], -viewPos[2], -viewPos[3])

	self.shader:use()
	--self.tex:enable()
	--self.tex:bind()
	gl.glCallList(self.displayList)
	--self.tex:unbind()
	--self.tex:disable()
	self.shader:useNone()

	local v = self.mesh[self.selectedVtx] or self.mesh[1]
	if v then
		gl.glColor3f(1,1,1)
		for sign=-1,1,2 do
			gl.glBegin(gl.GL_LINE_LOOP)
			for i=1,100 do
				local theta = 2 * math.pi * i / 100
				local radius = .1
				local x = v.dp_du * (math.cos(theta) * radius)
				local y = v.dp_dv * (math.sin(theta) * radius)
				local z = v.n * (.01 * sign)
				gl.glVertex3f((v.pt + x + y + z):unpack())
			end
			gl.glEnd()
		end
	end

	App.super.update(self)
end

App():run()
