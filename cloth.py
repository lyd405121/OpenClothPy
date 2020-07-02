import sys
import os
import taichi as ti
import time
import math
import numpy as np


ti.init(arch=ti.gpu)

imgSize = 720
screenRes = ti.Vector([imgSize, imgSize])
img = ti.Vector(3, dt=ti.f32, shape=[imgSize,imgSize])
depth = ti.var(dt=ti.f32, shape=[imgSize,imgSize])
gui = ti.GUI('Cloth', res=(imgSize,imgSize))


clothWid  = 4.0
clothHgt  = 4.0
clothResX = 31
clothResY = 31

pos_pre    = ti.Vector(3, dt=ti.f32, shape=(clothResX+1, clothResY+1))
pos        = ti.Vector(3, dt=ti.f32, shape=(clothResX+1, clothResY+1))
vel        = ti.Vector(3, dt=ti.f32, shape=(clothResX+1, clothResY+1))
F          = ti.Vector(3, dt=ti.f32, shape=(clothResX+1, clothResY+1))
J          = ti.Matrix(3, 3, dt=ti.f32, shape=(clothResX+1, clothResY+1))


eye        = ti.Vector(3, dt=ti.f32, shape=())
target     = ti.Vector(3, dt=ti.f32, shape=())
up         = ti.Vector(3, dt=ti.f32, shape=())
gravity    = ti.Vector(3, dt=ti.f32, shape=())
collisionC = ti.Vector(3, dt=ti.f32, shape=())

mass       = ti.var(dt=ti.i32, shape=())
damping    = ti.var(dt=ti.i32, shape=())
pointSize  = ti.var(dt=ti.i32, shape=())
           
deltaT     = ti.var(dt=ti.f32, shape=())
KsStruct   = ti.var(dt=ti.f32, shape=())
KdStruct   = ti.var(dt=ti.f32, shape=())
KsShear    = ti.var(dt=ti.f32, shape=())
KdShear    = ti.var(dt=ti.f32, shape=())
KsBend     = ti.var(dt=ti.f32, shape=())
KdBend     = ti.var(dt=ti.f32, shape=())
           
fov        = ti.var(dt=ti.f32, shape=())
near       = ti.var(dt=ti.f32, shape=())
far        = ti.var(dt=ti.f32, shape=())
collisionR = ti.var(dt=ti.f32, shape=())



@ti.func
def getNextNeighborKs(n):
    ks = 0.0
    if(n<4):
        ks = KsStruct
    else:
        if(n<8):
            ks = KsShear
        if(n<12) :
            ks = KsBend
    return ks
    
@ti.func
def getNextNeighborKd(n):
    kd = 0.0
    if(n<4):
        kd = KdStruct
    else:
        if(n<8):
            kd = KdShear
        else :
            kd = KdBend
    return kd
    
@ti.func
def getNextNeighborX(n):
    cood = 0
    if (n == 0) or (n == 4) or (n == 7):
        cood = 1
    if (n == 2) or (n == 5) or (n == 6):
        cood = -1
    if (n == 8):
        cood =  2
    if (n ==10):
        cood = -2
    return cood

@ti.func
def getNextNeighborY(n):
    cood = 0
    if (n == 1) or (n == 4) or (n == 5):
        cood = -1
    if (n == 3) or (n == 6) or (n == 7):
        cood = 1
    if (n == 9):
        cood = -2
    if (n ==11):
        cood = 2
    return cood
    
@ti.func
def get_length3(v):
    return ti.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)

@ti.func
def get_length2(v):
    return ti.sqrt(v.x*v.x+ v.y*v.y)


@ti.func
def get_proj(fovY, ratio, zn, zf):
    #  d3d perspective https://docs.microsoft.com/en-us/windows/win32/direct3d9/d3dxmatrixperspectiverh 
    # rember it is col major
    # xScale     0          0              0
    # 0        yScale       0              0
    # 0        0        zf/(zn-zf)        -1
    # 0        0        zn*zf/(zn-zf)      0
    # where:
    # yScale = cot(fovY/2)  
    # xScale = yScale / aspect ratio
    yScale = 1.0    / ti.tan(fovY/2)
    xScale = yScale / ratio
    return ti.Matrix([ [xScale, 0.0, 0.0, 0.0], [0.0, yScale, 0.0, 0.0], [0.0, 0.0, zf/(zn-zf), zn*zf/(zn-zf)], [0.0, 0.0, -1.0, 0.0] ])

@ti.func
def get_view(eye, target, up):
    #  d3d lookat https://docs.microsoft.com/en-us/windows/win32/direct3d9/d3dxmatrixlookatrh
    # rember it is col major
    #zaxis = normal(Eye - At)
    #xaxis = normal(cross(Up, zaxis))
    #yaxis = cross(zaxis, xaxis)
     
    # xaxis.x           yaxis.x           zaxis.x          0
    # xaxis.y           yaxis.y           zaxis.y          0
    # xaxis.z           yaxis.z           zaxis.z          0
    # dot(xaxis, eye)   dot(yaxis, eye)   dot(zaxis, eye)  1    ←  there is something wrong with it，  it should be '-'
    
    zaxis = eye - target
    zaxis = zaxis.normalized()
    xaxis = up.cross( zaxis)
    xaxis = xaxis.normalized()
    yaxis = zaxis.cross( xaxis)
    return ti.Matrix([ [xaxis.x, xaxis.y, xaxis.z, -xaxis.dot(eye)], [yaxis.x, yaxis.y, yaxis.z, -yaxis.dot(eye)], [zaxis.x, zaxis.y, zaxis.z, -zaxis.dot(eye)], [0.0, 0.0, 0.0, 1.0] ])

@ti.func
def transform(v):
    proj = get_proj(fov, 1.0, near, far)
    view = get_view(eye, target, up )
    
    screenP  = proj @ view @ ti.Vector([v.x, v.y, v.z, 1.0])
    screenP /= screenP.w
    
    return ti.Vector([(screenP.x+1.0)*0.5*screenRes.x, (screenP.y+1.0)*0.5*screenRes.y, screenP.z])
    

@ti.func
def fill_pixel(v, z, c):
    if (v.x >= 0) and  (v.x <screenRes.x) and (v.y >=0 ) and  (v.y < screenRes.y):
        if depth[v] >= z :
            img[v]   = c
            depth[v] = z
        
@ti.func
def draw_circle(v):
    v = transform(v)
    Centre = ti.Vector([ti.cast(v.x, ti.i32), ti.cast(v.y, ti.i32)])
    for i in range(-pointSize, pointSize+1):
        for j in range(-pointSize, pointSize+1):
            dis = ti.sqrt(i*i + j*j)
            if (dis < pointSize):
                fill_pixel(Centre+ti.Vector([i,j]), v.z, ti.Vector([1.0, 0.0, 0.0]))



#https://github.com/miloyip/line/blob/master/line_bresenham.c can be further optimized
@ti.func
def draw_line(v0,v1):
    v0 = transform(v0)
    v1 = transform(v1)
    
    s0 = ti.Vector([ti.cast(v0.x,  ti.i32), ti.cast(v0.y,  ti.i32)])
    s1 = ti.Vector([ti.cast(v1.x,  ti.i32), ti.cast(v1.y,  ti.i32)])
    dis = get_length2(s1 - s0)
    
    x0 = s0.x
    y0 = s0.y
    z0 = v0.z
    x1 = s1.x
    y1 = s1.y
    z1 = v1.z
    
    
    dx = abs(x1 - x0)
    sx = -1
    if x0 < x1 :
        sx = 1
    
    
    dy = abs(y1 - y0)
    sy = -1
    if y1 > y0:
        sy = 1
    
    dz = z1 - z0
    
    err = 0
    if dx > dy :
        err = ti.cast(dx/2,  ti.i32)
    else :
        err = ti.cast(-dy/2, ti.i32)
    
    for i in range(0, 64):
        distC = get_length2( ti.Vector([x1,y1])- ti.Vector([x0,y0]))
        
        fill_pixel(ti.Vector([x0,y0]), dz * (distC / dis) + v0.z, ti.Vector([0.0, 0.0, 1.0]))
        e2 = err
        if (e2 > -dx):
            err -= dy
            x0 += sx
        if (e2 <  dy):
            err += dx
            y0 += sy
        if (x0 == x1) and (y0 == y1):
            break

@ti.func
def SolveConjugateGradient(A, x, b):

    r = b-A@x
    d = r
    q = ti.Vector([0.0, 0.0, 0.0])
    alpha_new = 0.0
    alpha = 0.0
    beta  = 0.0
    delta_old = 0.0
    delta_new = r.dot(r)
    delta0    = delta_new
    
    for i in range(0,16):
        q = A@d
        alpha = delta_new/d.dot(q)
        x = x + alpha*d
        r = r - alpha*q
        delta_old = delta_new
        delta_new =  r.dot(r)
        beta = delta_new/delta_old
        d = r + beta*d
        i += 1
        if delta_new< 0.0000001:
            break
    return x   
    
@ti.kernel
def reset_cloth():
    for i, j in pos:
        pos[i, j]       = ti.Vector([clothWid * (i / clothResX) - clothWid / 2.0, 5.0, clothHgt * (j / clothResY)- clothHgt/2.0])
        pos_pre [i, j]  = pos[i, j]
        vel[i, j]       = ti.Vector([0.0, 0.0, 0.0])
        F[i, j]         = ti.Vector([0.0, 0.0, 0.0])


@ti.func
def compute_force(coord, jacobian):
    p1           = pos[coord]
    v1           = vel[coord]
    F[coord]     = gravity*mass + vel[coord]*damping
    E            = ti.Matrix.identity(ti.f32, 3)
    J[coord]     = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] )
    
    for k in range(0,12): 
        ks             = getNextNeighborKs(k)
        kd             = getNextNeighborKd(k)
        
        coord_offcet   = ti.Vector([getNextNeighborX(k), getNextNeighborY(k)])
        coord_neigh    = coord + coord_offcet
        
        
        if (coord_neigh.x >= 0) and (coord_neigh.x <= clothResX) and (coord_neigh.y >= 0) and (coord_neigh.y <= clothResY):
        
            rest_length = get_length2(coord_offcet * ti.Vector([clothWid / clothResX, clothHgt / clothResY]))
            
            p2          = pos[coord_neigh]
            v2          = (p2 - pos_pre[coord_neigh]) / deltaT
            
            deltaP      = p1 - p2
            deltaV      = v1 - v2 
            dist        = get_length3(deltaP)
            
            if jacobian > 0:
                dist2       = dist*dist
                lo_l        = rest_length/dist
                J[coord]   += ks*(lo_l * (E - deltaP.outer_product(deltaP) / dist2) - E)
    
            leftTerm    = -ks * (dist-rest_length)
            rightTerm   = kd * (deltaV.dot(deltaP)/dist)
            F[coord]   += deltaP.normalized() * (leftTerm + rightTerm) 		 			

@ti.func
def collision(coord):
    #collosoin
    if(pos[coord].y<0):
        pos[coord].y=0
        
    offcet = pos[coord] - collisionC
    dist   = get_length3(offcet)
    if(dist < collisionR):  
        delta0         = (collisionR - dist)
        pos[coord]    += offcet.normalized() *delta0
        pos_pre[coord] = pos[coord]
        vel[coord]     = ti.Vector([0.0, 0.0, 0.0])
        
    
@ti.kernel
def integrator_verlet():
    for i, j in pos:
        coord  = ti.Vector([i, j])
        compute_force(coord, 0)
        index  = j * (clothResX+1) + i
        
        if (index != 0) and (index != clothResX):
            
            collision(coord)
            
            acc = F[coord] / mass
            tmp             = pos[coord]       
            pos[coord]  = pos[coord] * 2.0 - pos_pre[coord] + acc * deltaT * deltaT
            vel[coord]  = (pos[coord] - pos_pre[coord]) /deltaT
            pos_pre[coord]  = tmp
            

@ti.kernel
def integrator_explicit():
    
    for i, j in pos:
        coord  = ti.Vector([i, j])
        compute_force(coord, 0)
        index  = j * (clothResX+1) + i
        
        if (index != 0) and (index != clothResX):
            collision(coord)
            
            tmp             = pos[coord]
            acc = F[coord] / mass
            pos[coord]  += vel[coord] * deltaT
            vel[coord]  += acc * deltaT
            pos_pre[coord]  = tmp            


        
@ti.kernel
def integrator_implicit():
    for i, j in pos:
        coord  = ti.Vector([i, j])
        compute_force(coord, 1)
        index  = j * (clothResX+1) + i
        
        if (index != 0) and (index != clothResX):
            collision(coord)
            tmp             = pos[coord]
            M     = ti.Matrix.identity(ti.f32, 3)*mass
            A     = M - deltaT * deltaT* J[coord]
            b     = M@vel[coord] + deltaT*F[coord]
            
            vel[coord] = SolveConjugateGradient(A, ti.Vector([0.0, 0.0, 0.0]), b)
            pos[coord]  += vel[coord] * deltaT
            pos_pre[coord]  = tmp   
            
      
@ti.kernel
def clear():
    for i, j in img:

        o    = eye
        c    = collisionC
        s    = ti.Vector([ (2.0 * i / screenRes.x - 1.0), (2.0 * j / screenRes.y - 1.0), 0.0, 1.0])

        proj = get_proj(fov, 1.0, near, far).inverse()
        view = get_view(eye, target, up ).inverse()
        s    = view @ proj @ s
        s    = s / s.w
        
        d    = ti.Vector([s.x - o.x, s.y - o.y, s.z - o.z])
        d    = d.normalized()
        #    h1    h2         -->two hitpoint
        # o--*--p--*--->d     -->Ray
        #   \   |
        #    \  |
        #     \ |
        #      c              -->circle centre
        oc = c - o
        oc_dist = get_length3(oc)
        op_dist = d.dot(oc)
        pc_dist = ti.sqrt(oc_dist*oc_dist -op_dist*op_dist )
        
        
        if pc_dist < collisionR:
            img[i, j]=ti.Vector([0, 1, 0])
            #h1 is nearer than h2
            # because h1 = o + t*d
            # so  |ch| = radius = |c - o - t*d| = |oc - td|
            # so  radius*radius = (oc - td)*(oc -td) = oc*oc +t*t*d*d -2*t*(oc*d)
            #so d*d*t^2   -2*(oc*d)* t + (oc*oc- radius*radius) = 0
            #cal ax^2+bx+c = 0
            
            aa = d.dot(d)
            bb = -2.0 * op_dist
            cc = oc.dot(oc) - collisionR*collisionR
            t1 = (-bb - ti.sqrt(bb * bb - 4.0 * aa * cc)) / 2.0 / aa
            h1 = o + t1 * d
            depth[i, j] = transform(h1).z
            
        else:
            img[i, j]=ti.Vector([0, 0, 0])
            depth[i, j] = 1.0
 
@ti.kernel
def draw_mass_point():
    for i, j in pos:
        draw_circle(pos[i, j])
        
@ti.kernel
def draw_grid():
    for i, j in pos:
        if i < clothResX:
            draw_line(pos[i, j], pos[i+1, j])
        if j < clothResY:
            draw_line(pos[i, j], pos[i, j+1])

 
pointSize  = 2       
eye        = ti.Vector([3.0, 3.0, 3.0])
target     = ti.Vector([0.0, 3.0, 0.0])
up         = ti.Vector([0.0, 1.0, 0.0])
gravity    = ti.Vector([0.0, -0.00981, 0.0])
collisionC = ti.Vector([0.0, 3.0, 0.0])
collisionR = 1.0
mass       = 1.0
deltaT     = 0.05
damping    = -0.0125
fov        = 1.5
near       = 1.0
far        = 1000.0
           
KsStruct   = 50.0
KdStruct   = -0.25
KsShear    = 50.0
KdShear    = -0.25
KsBend     = 50.0
KdBend     = -0.25

mode = 2
reset_cloth()
frame = 0

while gui.running:
    if gui.get_event(ti.GUI.ESCAPE) or (frame > 100):
        gui.running = False
    
    if gui.is_pressed('0', ti.GUI.LEFT):
        mode = 0
        reset_cloth() 
    
    if gui.is_pressed('1', ti.GUI.LEFT):
        mode = 1
        reset_cloth() 
        
    if gui.is_pressed('2', ti.GUI.LEFT):
        mode = 2
        reset_cloth() 
        
    #start_time = time.perf_counter()
    for i in  range(0, 20):
        if mode == 0:
            integrator_explicit()
        if mode == 1:
            integrator_implicit()
        if mode == 2:
            integrator_verlet()

    #end_time = time.perf_counter()
    #print("{:.4f}".format(end_time - start_time))
    
    clear()
    draw_mass_point()
    draw_grid()
    
    gui.set_image(img.to_numpy())
    gui.show()
    
    
    ti.imwrite(img, str(frame)+ ".png")
    frame += 1