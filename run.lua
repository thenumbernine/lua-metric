#!/usr/bin/env luajit
--[[

what to visualize?

*) Geodesics:

x^i_;j v^j = x^i_,j v^j + Gamma^i_jk x^j v^k = 0

for x -> u, v -> u, we get
u^i_;j u^j = u^i_,j u^j + Gamma^i_jk u^j u^k = 0

let du/ds = D_u u^i = u^i_;j u^j

x''^i + Gamma^i_jk x'^j x'^k = 0
x''^i + Gamma^i_jk x'^j x'^k = 0

*) connections - via two vectors, for a basis, in a direction, show Gamma^i_jk u^i v^j

*) geodesic deviation - [D_c, D_d] v^a = R^a_bcd v^b	<- choose vector v^a, then has freedom c & d
				<- or choose v^a, m^c, n^d, and show R^a_bcd v^b m^c n^d

*) Gaussian curvature at each point

*) Ricci curvature at each point, for a given direction u^a :  R_ab u^a u^b
--]]
require 'ext'
local ImGuiApp = require 'imguiapp'
local bit = require 'bit'
local ffi = require 'ffi'
local ig = require 'ffi.imgui'
local gl = require 'ffi.OpenGL'
local sdl = require 'ffi.sdl'
local GLProgram = require 'gl.program'
local GLTex2D = require 'gl.tex2d'
local FBO = require 'gl.fbo'
local Mouse = require 'gui.mouse'
local vec3 = require 'vec.vec3'
local quat = require 'vec.quat'
local matrix = require 'matrix'
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
		consts = {r=1},
		mins = {0, -math.pi},
		maxs = {math.pi, math.pi},
		step = {math.pi / 12, math.pi / 12},
		vars = {'theta', 'phi'},
		exprs = {
			'r * sin(theta) * cos(phi)',
			'r * sin(theta) * sin(phi)',
			'r * cos(theta)',
		},
	},
	{
		name = 'Polar',
		consts = {pi=math.pi},
		mins = {0, -math.pi},
		maxs = {2, math.pi},
		step = {1/4, math.pi/4},
		vars = {'r', 'theta'},
		exprs = {
			'r * cos(theta)',
			'r * sin(theta)',
			'0',
		},
	},
	{
		name = 'Torus',
		consts = {r=.25, R=1, pi=math.pi},
		mins = {-math.pi, -math.pi},
		maxs = {math.pi, math.pi},
		step = {math.pi / 4, math.pi / 12},
		vars = {'theta', 'phi'},
		exprs = {
			'(r * sin(theta) + R) * cos(phi)',
			'(r * sin(theta) + R) * sin(phi)',
			'r * cos(theta)',
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

	self.displayShader = GLProgram{
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

	self.pickShader = GLProgram{
		vertexCode = [[
varying vec2 intCoordV;
void main() {
	gl_Position = ftransform();
	intCoordV = gl_MultiTexCoord0.st;
}
]],
		fragmentCode = [[
varying vec2 intCoordV;
void main() {
	gl_FragColor = vec4(intCoordV, 0., 1.);
}
]],
	}

	self.floatTex = GLTex2D{
		width = 2048,
		height = 2048,
		internalFormat = gl.GL_RGBA32F,
		format = gl.GL_RGBA,
		type = gl.GL_FLOAT,
		minFilter = gl.GL_NEAREST,
		magFilter = gl.GL_NEAREST,
	}
	self.fbo = FBO{
		width = self.floatTex.width, 
		height = self.floatTex.height, 
		useDepth = true,
	}
	self.fbo:setColorAttachmentTex2D(0, self.floatTex.id)
	self.fbo:bind()
	self.fbo:check()
	self.fbo:unbind()

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

	local function simplifyTrig(x)
		x = x:map(function(expr)
			-- I need to make a symmath operation out of this 
			-- pow div -> mul div
			-- (x/2)^2 -> x^2 / 2^2
			if symmath.powOp.is(expr)
			and symmath.divOp.is(expr[1])
			then
				if expr[2].value == 0 then
					return 1
				else
					local num = (symmath.clone(expr[1][1])^symmath.clone(expr[2]))()
					local denom = (symmath.clone(expr[1][2])^symmath.clone(expr[2]))() 
					return (num / denom)()
				end
			end
		end)
		return x:map(function(expr)
			if symmath.powOp.is(expr)
			and expr[2] == symmath.Constant(2)
			and symmath.cos.is(expr[1])
			then
				return 1 - symmath.sin(expr[1][1]:clone())^2
			end
		end)()
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

		local p = Tensor('^I', eqns:map(function(eqn) return eqn.expr end):unpack())
		local e = Tensor'_u^I'
		e['_u^I'] = p'^I_,u'()
		local g = (e'_u^I' * e'_v^J' * eta'_IJ')()
print('g before', g)
		g = simplifyTrig(g)
print('g simplified', g)
		Tensor.metric(g)
		local dg = Tensor'_uvw'
		dg['_uvw'] = g'_uv,w'()
print('dg before', dg)	
		dg = simplifyTrig(dg)
print('dg simplified', dg)	
		local Gamma = ((dg'_uvw' + dg'_uwv' - dg'_vwu')/2)()
		Gamma = Gamma'^u_vw'()
print('Gamma before', Gamma)	
		Gamma = simplifyTrig(Gamma)
print('Gamma simplified', Gamma)	
		
		-- [[	
		local dGamma = Tensor'^a_bcd'
		dGamma['^a_bcd'] = Gamma'^a_bc,d'()
		local Riemann = Tensor'^a_bcd'
		Riemann['^a_bcd'] = (dGamma'^a_bdc' - dGamma'^a_bcd' + Gamma'^a_uc' * Gamma'^u_bd' - Gamma'^a_ud' * Gamma'^u_bc')()
		local Ricci = Tensor'_ab'
		Ricci['_ab'] = Riemann'^c_acb'()
		local Gaussian = Ricci'^a_a'()
		--]]
		
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
		for i,xi in ipairs(vars) do
			for j,xj in ipairs(vars) do
				for k,xk in ipairs(vars) do
					for l,xl in ipairs(vars) do
						if Riemann[i][j][k][l] ~= symmath.Constant(0) then
							self.strs:insert('R^'..xi..'_'..xj..'_'..xk..'_'..xl..' = '..Riemann[i][j][k][l])
						end
					end
				end
			end
		end
		for _,str in ipairs(self.strs) do
			print(str)
		end
	
		self.Gamma = range(2):map(function(i)
			return range(2):map(function(j)
				return range(2):map(function(k)
					local expr = symmath.clone(Gamma[i][j][k])()
					for var,value in pairs(self.consts) do
						expr = expr:replace(var, value)()
					end
					return (expr:compile(vars))
				end)
			end)
		end)
	end

	self.displayList = gl.glGenLists(1)
	gl.glNewList(self.displayList, gl.GL_COMPILE)

	self.getpt = function(u1value, u2value)
		local pt = vec3()
		for i=1,3 do
			if eqns[i] then 
				pt[i] = eqns[i].func(u1value, u2value) 
			end
		end
		return pt
	end
	
	local u1, u2 = params:unpack()

	self.get_dp_du1 = function(u1value, u2value)
		local du = (u1.max - u1.min) / u1.divs
		return ((self.getpt(u1value + du, u2value) - self.getpt(u1value - du, u2value))):normalize()
	end

	self.get_dp_du2 = function(u1value, u2value)
		local dv = (u2.max - u2.min) / u2.divs
		return ((self.getpt(u1value, u2value + dv) - self.getpt(u1value, u2value - dv))):normalize()
	end

	self.get_dp_du = function(...)
		return matrix{
			matrix{self.get_dp_du1(...):unpack()},
			matrix{self.get_dp_du2(...):unpack()},
		}:transpose()
	end

	for u1div=0,u1.divs-1 do
		gl.glBegin(gl.GL_TRIANGLE_STRIP)
		for u2div=0,u2.divs do
			local u2frac = u2div / u2.divs
			local u2value = u2frac * (u2.max - u2.min) + u2.min
			for u1ofs=1,0,-1 do
				local u1frac = (u1div + u1ofs) / u1.divs
				local u1value = u1frac * (u1.max - u1.min) + u1.min
				local pt = self.getpt(u1value, u2value)
				gl.glTexCoord2f(u1value, u2value)
			
				local dp_du = self.get_dp_du1(u1value, u2value)
				local dp_dv = self.get_dp_du2(u1value, u2value) 
				local n = vec3.cross(dp_du, dp_dv):normalize()
				gl.glNormal3f(n:unpack())
				
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
	self.selectedPt = (vec2(table.unpack(option.mins)) + option.maxs) * .5
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

function App:getCoord(mx,my)
	local results = ffi.new'float[4]'
	local depth = ffi.new'float[1]'
	gl.glViewport(0, 0, self.fbo.width, self.fbo.height)
	self.fbo:bind()
	self.fbo:check()
	gl.glClear(bit.bor(gl.GL_DEPTH_BUFFER_BIT, gl.GL_COLOR_BUFFER_BIT))
	gl.glMatrixMode(gl.GL_PROJECTION)
	gl.glLoadIdentity()
	local ar = self.width / self.height
	gl.glFrustum(-zNear * ar * tanFovX, zNear * ar * tanFovX, -zNear * tanFovY, zNear * tanFovY, zNear, zFar);
	self:drawMesh'pick'
	local ix = math.floor(self.fbo.width * mx)
	local iy = math.floor(self.fbo.height * my)

	local u
	if ix >= 0 and iy >= 0 and ix < self.fbo.width and iy < self.fbo.height then
		gl.glReadPixels(ix, iy, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, depth)
		if depth[0] < 1 then 
			gl.glReadPixels(ix, iy, 1, 1, gl.GL_RGBA, gl.GL_FLOAT, results)
			u = vec2(results[0],results[1])
		end
	end
	self.fbo:unbind()
	return u
end

function App:update()		
	mouse:update()
	
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
			if mouse.leftDown then
				self.selectedPt = self:getCoord(mouse.pos:unpack()) or self.selectedPt
			end	
		elseif self.controlPtr[0] == controlIndexes.direct-1 then
			if mouse.leftDown
			and self.selectedPt
			then
				local u = self:getCoord(mouse.pos:unpack())
				if u then
					local dir = self.getpt(u:unpack()) - self.getpt(self.selectedPt:unpack())
					local dp_du = self.get_dp_du(u:unpack()):transpose()
					dir = matrix{dir}:transpose()
					local udir = dp_du * dir
					udir = udir:transpose()[1]
					self.dir = vec2(table.unpack(udir)):normalize()
				end	
			end	
		end
	end
		
	viewPos = viewAngle:zAxis() * viewDist 

	gl.glViewport(0, 0, self.width, self.height)
	gl.glClear(bit.bor(gl.GL_DEPTH_BUFFER_BIT, gl.GL_COLOR_BUFFER_BIT))
	gl.glMatrixMode(gl.GL_PROJECTION)
	gl.glLoadIdentity()
	local ar = self.width / self.height
	gl.glFrustum(-zNear * ar * tanFovX, zNear * ar * tanFovX, -zNear * tanFovY, zNear * tanFovY, zNear, zFar);

	self:drawMesh'display'


	if self.selectedPt then
		-- draw selected coordinate
		local pt = self.getpt(self.selectedPt:unpack())
		local dp_du = self.get_dp_du1(self.selectedPt:unpack())
		local dp_dv = self.get_dp_du2(self.selectedPt:unpack())
		local n = vec3.cross(dp_du, dp_dv)
		gl.glColor3f(1,1,1)
		for sign=-1,1,2 do
			gl.glBegin(gl.GL_LINE_LOOP)
			for i=1,100 do
				local theta = 2 * math.pi * i / 100
				local radius = .1
				local x = dp_du * (math.cos(theta) * radius)
				local y = dp_dv * (math.sin(theta) * radius)
				local z = n * (.01 * sign)
				gl.glVertex3f((pt + x + y + z):unpack())
			end
			gl.glEnd()
		end
	end
	
	gl.glDisable(gl.GL_DEPTH_TEST)
	
	if self.dir then
		gl.glColor3f(1,1,0)
		gl.glBegin(gl.GL_LINE_STRIP)
		local u = self.selectedPt
		local du_dl = self.dir
		local l = .05
		for i=1,100 do
			u = u + du_dl * l
			gl.glVertex3d(self.getpt(u:unpack()):unpack())
		end
		gl.glEnd()
	
		gl.glColor3f(1,0,1)
		gl.glBegin(gl.GL_LINE_STRIP)
		local u = self.selectedPt
		local du_dl = self.dir
		local dl = .05
		for iter=1,100 do
			-- x''^a + Gamma^a_uv x'^u x'^v = 0
			-- x''^a = -Gamma^a_uv x'^u x'^v
			u = u + du_dl * dl
			for i=1,2 do
				for j=1,2 do
					for k=1,2 do
						du_dl[i] = du_dl[i] - self.Gamma[i][j][k](u[1], u[2]) * du_dl[j] * du_dl[k] * dl
					end
				end
			end
			gl.glVertex3d(self.getpt(u:unpack()):unpack())
		end
		gl.glEnd()
	end
	
	gl.glEnable(gl.GL_DEPTH_TEST)

	App.super.update(self)
end

function App:drawMesh(method)
	gl.glMatrixMode(gl.GL_MODELVIEW)
	gl.glLoadIdentity()
	gl.glScaled(viewScale, viewScale, viewScale)
	local aa = viewAngle:toAngleAxis()
	gl.glRotated(-aa[4], aa[1], aa[2], aa[3])
	gl.glTranslated(-viewPos[1], -viewPos[2], -viewPos[3])

	if method == 'display' then
		self.displayShader:use()
	elseif method == 'pick' then
		self.pickShader:use()
	end
	gl.glCallList(self.displayList)
	GLProgram:useNone()
end

App():run()
