#!/usr/bin/env luajit
--[[

what to visualize?

*) partial derivatives: show chart^I(x^b + lambda v^b)

*) Geodesics:

x^i_;j v^j = x^i_,j v^j + Gamma^i_jk x^j v^k = 0

for x -> u, v -> u, we get
u^i_;j u^j = u^i_,j u^j + Gamma^i_jk u^j u^k = 0

let du/ds = D_u u^i = u^i_;j u^j

x''^i = -Gamma^i_jk x'^j x'^k

*) connections - via two vectors, for a basis, in a direction, show Gamma^i_jk u^i v^j

TODO:
*) geodesic deviation - [D_c, D_d] v^a = R^a_bcd v^b	<- choose vector v^a, then has freedom c & d
				<- or choose v^a, m^c, n^d, and show R^a_bcd v^b m^c n^d

*) Gaussian curvature at each point

*) Ricci curvature at each point, for a given direction u^a :  R_ab u^a u^b

TODO:
*) Ricci curvature at each point, for two given directions: R_ab u^a u^b

*) parallel propagation between two points
--]]
require 'ext'
local bit = require 'bit'
local ffi = require 'ffi'
local ig = require 'ffi.imgui'
local gl = require 'gl'
local sdl = require 'ffi.sdl'
local GLProgram = require 'gl.program'
local GLTex2D = require 'gl.tex2d'
local GradientTex = require 'gl.gradienttex'
local glreport = require 'gl.report'
local FBO = require 'gl.fbo'
local matrix = require 'matrix'
local symmath = require 'symmath'
symmath.tostring = require 'symmath.export.SingleLine'

function matrix:normalize()
	return self / self:norm()
end

-- cross product, but with matrix instead of vector
function cross(a,b)
	return matrix{
		a[2]*b[3] - a[3]*b[2],
		a[3]*b[1] - a[1]*b[3],
		a[1]*b[2] - a[2]*b[1],
	}
end

local View = require 'glapp.view'

local App = class(require 'glapp.orbit'(require 'imguiapp'))
App.viewDist = 3
App.title = 'Metric Visualization'

local eqnPtr = ffi.new('int[1]', 0)

--[[
TODO higher dimension parameters: u,v,w ... what about 4d?  stuv?
and how to display this?  as a mesh
and how to click-to-interact with this?
--]]
local params = table{
	{var=symmath.var'u', divs=300, min=-5, max=5},
	{var=symmath.var'v', divs=300, min=-5, max=5},
}

local u, v = params[1].var, params[2].var

--[[
TODO higher dimensions here?
--]]
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
		mins = {-5, -5},
		maxs = {5, 5},
		step = {1, 1},
		exprs = {'u', 'v', '0'},
	},
	{
		name = 'Sphere',
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
		step = {.2, .2},
		vars = {'u', 'v'},
		exprs = {'u', 'v', '-.5 * (u^2 + v^2)'},
	},
	{
		name = 'Hyperboloid',
		consts = {},
		mins = {-1, -1},
		maxs = {1, 1},
		step = {.2, .2},
		vars = {'u', 'v'},
		exprs = {'u', 'v', '.5 * (u^2 - v^2)'},
	},
}

local controls = table{'rotate', 'select', 'direct'}
local controlIndexes = controls:map(function(v,k) return k,v end)

local displays = table{'grid', 'Gaussian', 'Ricci'}
local displayIndexes = displays:map(function(v,k) return k,v end)

function App:init(...)
	App.super.init(self, ...)

	self.dir = matrix{0,0}
	self.consts = {}
end

function App:initGL(...)
	App.super.initGL(self, ...)

	gl.glClearColor(1,1,1,1)
	gl.glEnable(gl.GL_DEPTH_TEST)

	self.gridShader = GLProgram{
		vertexCode = [[
varying vec2 gridCoord;
varying vec3 normalV;
varying vec3 vertexV;
void main() {
	normalV = (gl_ModelViewMatrix * vec4(gl_Normal, 0.)).xyz;
	vec4 mvtx = gl_ModelViewMatrix * gl_Vertex;
	vertexV = mvtx.xyz;
	gl_Position = gl_ProjectionMatrix * mvtx;
	gridCoord = gl_MultiTexCoord0.st;
}
]],
		fragmentCode = [[
varying vec2 gridCoord;
varying vec3 normalV;
varying vec3 vertexV;
uniform vec2 step;
void main() {
	vec3 n = normalize(normalV);
	if (n.z < 0.) n = -n;	//backface lighting
	
	vec2 fc = mod(gridCoord.xy / step, 1.); //grid
	float i = 1. - 8. * fc.x * fc.y * (1. - fc.x) * (1. - fc.y);
	i = pow(i, 50.);
	
	//gl_FragColor = vec4(.25, .5, .5, 1.);
	gl_FragColor = vec4(1,1,1,1);
	gl_FragColor.rgb *= 1. - i;
	vec3 u = normalize(vertexV);
	float l = dot(n, u);
	gl_FragColor.rgb *= max(abs(l), .3);
}
]],
	}

	self.pickShader = GLProgram{
		vertexCode = [[
varying vec2 gridCoord;
void main() {
	gl_Position = ftransform();
	gridCoord = gl_MultiTexCoord0.st;
}
]],
		fragmentCode = [[
varying vec2 gridCoord;
void main() {
	gl_FragColor = vec4(gridCoord, 0., 1.);
}
]],
	}

	self.gradientShader = GLProgram{
		vertexCode = [[
varying vec2 gridCoord;
void main() {
	gl_Position = ftransform();
	gridCoord = gl_MultiTexCoord0.st;
}
]],
		fragmentCode = [[
varying vec2 gridCoord;
uniform sampler2D tex;
uniform sampler2D gradientTex;
uniform vec2 mins, maxs;
void main() {
	vec2 tc = (gridCoord - mins) / (maxs - mins);
	float value = texture2D(tex, tc).r;
	gl_FragColor = texture2D(gradientTex, vec2(value, .5));
}
]],
		uniforms = {
			tex = 0,
			gradientTex = 1,
		},
	}

	local hsvWidth = 256
	self.gradientTex = GradientTex(hsvWidth,
--[[ rainbow or heatmap or whatever
		{
			{0,0,0,0},
			{1,0,0,1/6},
			{1,1,0,2/6},
			{0,1,1,3/6},
			{0,0,1,4/6},
			{1,0,1,5/6},
			{0,0,0,6/6},
		},
--]]
-- [[ sunset pic from https://blog.graphiq.com/finding-the-right-color-palettes-for-data-visualizations-fcd4e707a283#.inyxk2q43
		table{
			{22,31,86},
			{34,54,152},
			{87,49,108},
			{156,48,72},
			{220,60,57},
			{254,96,50},
			{255,188,46},
			{255,255,55},
		}:map(function(c,i)
			return table(matrix(c)/255):append{1}
		end),
--]]
		false)
	-- change to 2D so imgui can use it
	local data = ffi.new('unsigned char[?]', hsvWidth*4)
	gl.glGetTexImage(gl.GL_TEXTURE_1D, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
	self.gradientTex:unbind()
	self.gradientTex = GLTex2D{
		internalFormat = gl.GL_RGBA,
		width = hsvWidth,
		height = 1,
		format = gl.GL_RGBA,
		type = gl.GL_UNSIGNED_BYTE,
		data = data,
		minFilter = gl.GL_LINEAR_MIPMAP_LINEAR,
		magFilter = gl.GL_LINEAR,
		wrap = {
			s = gl.GL_CLAMP_TO_EDGE,
			t = gl.GL_REPEAT,
		},
		generateMipmap = true,
	}


	local function makeFloatTex()
		return GLTex2D{
			width = 256,
			height = 256,
			internalFormat = gl.GL_RGBA32F,
			format = gl.GL_RGBA,
			type = gl.GL_FLOAT,
			minFilter = gl.GL_NEAREST,
			magFilter = gl.GL_NEAREST,
		}
	end

	self.floatTex = makeFloatTex() 
	self.fbo = FBO{
		width = self.floatTex.width, 
		height = self.floatTex.height, 
		useDepth = true,
	}
	self.fbo:setColorAttachmentTex2D(0, self.floatTex.id)
	self.fbo:bind()
	assert(self.fbo:check())
	self.fbo:unbind()

	self.ricciTex = makeFloatTex()
	self.gaussianTex = makeFloatTex()

	self:setOption(options[1])
	
	self.controlPtr = ffi.new('int[1]', controlIndexes.rotate)
	self.displayPtr = ffi.new('int[1]', displayIndexes.grid)
end

function App:calculateMesh()
	
	if self.displayList then
		gl.glDeleteLists(self.displayList, 1)
	end

	-- refresh ...
	
	local vars = params:map(function(param) return param.var end)

	local function compileWithConsts(expr)
		expr = symmath.clone(expr)()
		for var,value in pairs(self.consts) do
			expr = expr:replace(var, value)()
		end
		return (expr:compile(vars))
	end

	for _,eqn in ipairs(eqns) do
		eqn.func = compileWithConsts(eqn.expr)
	end

	do
		local x,y,z = symmath.vars('x','y','z')
		local flatCoords = {x,y,z}
		local curvedCoords = vars
		local Tensor = symmath.Tensor
		local flatChart = Tensor.Chart{coords=flatCoords, symbols='IJKLMN'}
		local chart = Tensor.Chart{coords=curvedCoords}
		local eta = Tensor('_IJ', {1,0,0},{0,1,0},{0,0,1})
		flatChart:setMetric(eta, eta, 'I')

		local p = Tensor('^I', eqns:map(function(eqn) return eqn.expr end):unpack())
		local e = Tensor'_u^I'
		e['_u^I'] = p'^I_,u'()
		local g = (e'_u^I' * e'_v^J' * eta'_IJ')()
		chart:setMetric(g)
		local dg = Tensor'_uvw'
		dg['_uvw'] = g'_uv,w'()
		local Gamma = ((dg'_uvw' + dg'_uwv' - dg'_vwu')/2)()
		Gamma = Gamma'^u_vw'()
		
		local dGamma = Tensor'^a_bcd'
		dGamma['^a_bcd'] = Gamma'^a_bc,d'()
		local Riemann = Tensor'^a_bcd'
		Riemann['^a_bcd'] = (dGamma'^a_bdc' - dGamma'^a_bcd' + Gamma'^a_uc' * Gamma'^u_bd' - Gamma'^a_ud' * Gamma'^u_bc')()
		local Ricci = Tensor'_ab'
		Ricci['_ab'] = Riemann'^c_acb'()
		local Gaussian = Ricci'^a_a'()

		local function addStrs(name, expr)
			if symmath.Tensor:isa(expr) then
				local any
				for k,v in expr:iter() do
					if v ~= symmath.Constant(0) then
						local i = table.map(expr.variance, function(v,i)
							return v.lower and '_'..k[i] or '^'..k[i]
						end):concat()
						self.strs:insert(name..i..' = '..v)
						any = true
					end
				end
				if not any then
					local i = table.map(expr.variance, tostring):concat()
					self.strs:insert(name..i..' = 0')
				end
			else
				self.strs:insert(name..' = '..expr)
			end
		end
		
		self.strs = table()
		addStrs('g', g)
		addStrs('Gamma', Gamma)
		addStrs('R', Riemann)
		addStrs('R', Ricci)
		addStrs('R', Gaussian)
		for _,str in ipairs(self.strs) do
			print(str)
		end

		local func_Gamma = compileWithConsts(Gamma)
		self.Gamma = function(...)
			return matrix(func_Gamma(...))
		end

		local func_Ricci = compileWithConsts(Ricci)
		self.Ricci = function(...) 
			return matrix(func_Ricci(...)) 
		end

		self.Gaussian = compileWithConsts(Gaussian)
	end

	self.displayList = gl.glGenLists(1)
	gl.glNewList(self.displayList, gl.GL_COMPILE)

	self.getpt = function(u1value, u2value)
		local pt = matrix()
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
			self.get_dp_du1(...),
			self.get_dp_du2(...),
		}
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
				local n = cross(dp_du, dp_dv):normalize()
				gl.glNormal3f(n:unpack())
				
				gl.glVertex3f(pt:unpack())
			end
		end
		gl.glEnd()
	end
	
	gl.glEndList()				

	local buf = ffi.new('float[?]', self.gaussianTex.width * self.gaussianTex.height * 4)
	
	local gaussianMin = math.huge
	local gaussianMax = -math.huge
	for i=0,self.gaussianTex.width-1 do
		local u1value = (i+.5) / self.gaussianTex.width * (u1.max - u1.min) + u1.min
		for j=0,self.gaussianTex.height-1 do
			local index = 0 + 4 * (i + self.gaussianTex.width * j)
			local u2value = (j+.5) / self.gaussianTex.height * (u2.max - u2.min) + u2.min
			local R = self.Gaussian(u1value, u2value)
			gaussianMin = math.min(gaussianMin, R)
			gaussianMax = math.max(gaussianMax, R)
			buf[index] = R
		end
	end
	if gaussianMax - gaussianMin < 1e-10 then
		gaussianMax = gaussianMin + 1e-10
	end
	for i=0,self.gaussianTex.width-1 do
		for j=0,self.gaussianTex.height-1 do
			local index = 0 + 4 * (i + self.gaussianTex.width * j)
			buf[index] = (buf[index] - gaussianMin) / (gaussianMax - gaussianMin)
		end
	end

	self.gaussianTex:bind()
	gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.gaussianTex.width, self.gaussianTex.height, gl.GL_RGBA, gl.GL_FLOAT, buf)
	self.gaussianTex:unbind()
	glreport'here'

	self:updateRicciTex()
end

function App:updateRicciTex()
	local u1, u2 = params:unpack()
	local buf = ffi.new('float[?]', self.gaussianTex.width * self.gaussianTex.height * 4)
	local ricciMin = math.huge
	local ricciMax = -math.huge
	for i=0,self.ricciTex.width-1 do
		local u1value = (i+.5) / self.ricciTex.width * (u1.max - u1.min) + u1.min
		for j=0,self.ricciTex.height-1 do
			local index = 0 + 4 * (i + self.ricciTex.width * j)
			local u2value = (j+.5) / self.ricciTex.height * (u2.max - u2.min) + u2.min
			local R = self.Ricci(u1value, u2value) * self.dir * self.dir
			ricciMin = math.min(ricciMin, R)
			ricciMax = math.max(ricciMax, R)
			buf[index] = R
		end
	end
	if ricciMax - ricciMin < 1e-10 then
		ricciMax = ricciMin + 1e-10
	end
	for i=0,self.ricciTex.width-1 do
		for j=0,self.ricciTex.height-1 do
			local index = 0 + 4 * (i + self.ricciTex.width * j)
			buf[index] = (buf[index] - ricciMin) / (ricciMax - ricciMin)
		end
	end

	self.ricciTex:bind()
	gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.ricciTex.width, self.ricciTex.height, gl.GL_RGBA, gl.GL_FLOAT, buf)
	self.ricciTex:unbind()
	glreport'here'

end

function App:event(event, ...)
	-- TODO disable orbit
	if self.controlPtr[0] ~= controlIndexes.rotate then
		pushView = View(self.view)
	end
	if App.super.event then
		App.super.event(self, event, ...)
	end
	if self.controlPtr[0] ~= controlIndexes.rotate then
		self.view = pushView
	end
end

function App:setOption(option)
	self.consts = table.map(option.consts, function(v,k)
		return v, symmath.var(k)
	end)
	for i=1,#option.vars do
		params[i].min = option.mins[i]
		params[i].max = option.maxs[i]
		params[i].step = option.step[i]
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
	self.selectedPt = (matrix(option.mins) + matrix(option.maxs)) * .5
end

function App:updateGUI()
	if ig.igCollapsingHeader'controls:' then
		for i,control in ipairs(controls) do	
			ig.igRadioButton_IntPtr(control, self.controlPtr, i)
		end
	end
	if ig.igCollapsingHeader'display' then
		for i,display in ipairs(displays) do
			ig.igRadioButton_IntPtr(display, self.displayPtr, i)
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
			ig.igPushID_Str('parameter '..i)
			--[[ 2D only for now
			if ig.igButton'-' then
				remove = remove or table()
				remove:insert(i)
			end
			--]]
			ig.igSameLine()
			ig.igText(param.var.name)
			local int = ffi.new('int[1]', param.divs)
			if ig.igInputInt('divs', int, 1, 100, ig.ImGuiInputTextFlags_EnterReturnsTrue) then
				param.divs = int[0]
			end
			local float = ffi.new('float[1]', param.min)
			if ig.igInputFloat('min', float, 0, 0, '%.3f', ig.ImGuiInputTextFlags_EnterReturnsTrue) then
				param.min = float[0]
			end
			float[0] = param.max
			if ig.igInputFloat('max', float, 0, 0, '%.3f', ig.ImGuiInputTextFlags_EnterReturnsTrue) then
				param.max = float[0]
			end
			ig.igPopID()
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
			ig.igPushID_Str('equation '..i)
			if ig.igButton'-' then
				remove = remove or table()
				remove:insert(i)
			end
			ig.igSameLine()
			ig.igText(eqn.name..' = '..eqn.expr)
			ig.igPopID()
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
	assert(self.fbo:check())
	gl.glClear(bit.bor(gl.GL_DEPTH_BUFFER_BIT, gl.GL_COLOR_BUFFER_BIT))
	self:drawMesh'pick'
	local ix = math.floor(self.fbo.width * mx)
	local iy = math.floor(self.fbo.height * my)

	local u
	if ix >= 0 and iy >= 0 and ix < self.fbo.width and iy < self.fbo.height then
		gl.glReadPixels(ix, iy, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, depth)
		if depth[0] < 1 then 
			gl.glReadPixels(ix, iy, 1, 1, gl.GL_RGBA, gl.GL_FLOAT, results)
			u = matrix{tonumber(results[0]),tonumber(results[1])}
		end
	end
	self.fbo:unbind()
	gl.glViewport(0, 0, self.width, self.height)
	return u
end

function App:update()		
	local canHandleMouse = not ig.igGetIO()[0].WantCaptureMouse
	local canHandleKeyboard = not ig.igGetIO()[0].WantCaptureKeyboard
	
	if canHandleMouse then 
		if self.controlPtr[0] == controlIndexes.select then
			if self.mouse.leftDown then
				self.selectedPt = self:getCoord(self.mouse.pos:unpack()) or self.selectedPt
			end	
		elseif self.controlPtr[0] == controlIndexes.direct then
			if self.mouse.leftDown
			and self.selectedPt
			then
				local u = self:getCoord(self.mouse.pos:unpack())
				if u then
					local dir = self.getpt(u:unpack()) - self.getpt(self.selectedPt:unpack())
					local dp_du = self.get_dp_du(u:unpack())
					local udir = dp_du * dir
					self.dir = udir:normalize()
					-- if we change dir and are showing ricci curvature then update the mesh
					-- TODO do this in GPU
					-- but that means finding the min/max in GPU as well
					-- which isn't so tough ... just do a FBO reduce (might be easier in OpenCL)
					if self.displayPtr[0] == displayIndexes.Ricci then
						self:updateRicciTex()
					end
				end
			end	
		end
	end
	
	gl.glClear(bit.bor(gl.GL_COLOR_BUFFER_BIT, gl.GL_DEPTH_BUFFER_BIT))
	self.view:setupProjection(self.width / self.height)
	self:drawMesh'display'

	if self.selectedPt then
		-- draw selected coordinate
		local pt = self.getpt(self.selectedPt:unpack())
		local dp_du = self.get_dp_du1(self.selectedPt:unpack())
		local dp_dv = self.get_dp_du2(self.selectedPt:unpack())
		local n = cross(dp_du, dp_dv)
		gl.glColor3f(1,0,0)
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
	
	if self.dir then
		gl.glColor3f(.75,.5,0)
		for sign=-1,1,2 do
			local u = self.selectedPt
			local du_dl = matrix(self.dir)
			local l = .05
			gl.glBegin(gl.GL_LINE_STRIP)
			for i=1,100 do
				u = u + du_dl * l
				local dp_du = self.get_dp_du1(u:unpack())
				local dp_dv = self.get_dp_du2(u:unpack())
				local n = cross(dp_du, dp_dv)
				gl.glVertex3d((self.getpt(u:unpack()) + n * (.01 * sign)):unpack())
			end
			gl.glEnd()
		end

		gl.glColor3f(.5,0,.75)
		for sign=-1,1,2 do
			local u = self.selectedPt
			local du_dl = matrix(self.dir)
			local dl = .05
			gl.glBegin(gl.GL_LINE_STRIP)
			for iter=1,100 do
				-- x''^a + Gamma^a_uv x'^u x'^v = 0
				-- x''^a = -Gamma^a_uv x'^u x'^v
				u = u + du_dl * dl
				du_dl = du_dl - self.Gamma(u:unpack()) * du_dl * du_dl * dl
				local dp_du = self.get_dp_du1(u:unpack())
				local dp_dv = self.get_dp_du2(u:unpack())
				local n = cross(dp_du, dp_dv)
				gl.glVertex3d((self.getpt(u:unpack()) + n * (.01 * sign)):unpack())
			end
			gl.glEnd()
		end
	end

	App.super.update(self)

	glreport'here'
end

function App:drawMesh(method)
	self.view:setupModelView()
	if method == 'display' then
		if self.displayPtr[0] == displayIndexes.grid then 
			self.gridShader:use()
			gl.glUniform2f(self.gridShader.uniforms.step.loc, params[1].step, params[2].step)
		elseif self.displayPtr[0] == displayIndexes.Gaussian then
			self.gradientShader:use()
			self.gaussianTex:bind(0)
			self.gradientTex:bind(1)
			gl.glUniform2f(self.gradientShader.uniforms.mins.loc, params[1].min, params[2].min)
			gl.glUniform2f(self.gradientShader.uniforms.maxs.loc, params[1].max, params[2].max)
		elseif self.displayPtr[0] == displayIndexes.Ricci then
			self.gradientShader:use()
			self.ricciTex:bind(0)
			self.gradientTex:bind(1)
			gl.glUniform2f(self.gradientShader.uniforms.mins.loc, params[1].min, params[2].min)
			gl.glUniform2f(self.gradientShader.uniforms.maxs.loc, params[1].max, params[2].max)
		end
	elseif method == 'pick' then
		self.pickShader:use()
	end
	gl.glCallList(self.displayList)
	GLTex2D:unbind(1)
	GLTex2D:unbind(0)
	GLProgram:useNone()
end

App():run()
