import sys
import os
import taichi as ti
import time
import math
import numpy as np


ti.init(arch=ti.cpu)

#gui system using taichi-ggui:
#https://docs.taichi.graphics/zh-Hans/docs/lang/articles/misc/ggui
imgSize = 512


clothWid  = 4.0
clothHgt  = 4.0
clothResX = 31

num_triangles = clothResX  * clothResX * 2
indices       = ti.field(int, num_triangles * 3)
vertices      = ti.Vector.field(3, float, (clothResX+1)*(clothResX+1))

pos_pre    = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX+1, clothResX+1))
pos        = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX+1, clothResX+1))
vel        = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX+1, clothResX+1))
F          = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX+1, clothResX+1))
J          = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(clothResX+1, clothResX+1))




           

KsStruct   = ti.field(dtype=ti.f32, shape=())
KdStruct   = ti.field(dtype=ti.f32, shape=())
KsShear    = ti.field(dtype=ti.f32, shape=())
KdShear    = ti.field(dtype=ti.f32, shape=())
KsBend     = ti.field(dtype=ti.f32, shape=())
KdBend     = ti.field(dtype=ti.f32, shape=())
           

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
        pos[i, j]       = ti.Vector([clothWid * (i / clothResX) - clothWid / 2.0, 5.0, clothHgt * (j / clothResX)- clothHgt/2.0])
        pos_pre [i, j]  = pos[i, j]
        vel[i, j]       = ti.Vector([0.0, 0.0, 0.0])
        F[i, j]         = ti.Vector([0.0, 0.0, 0.0])


        if i < clothResX - 1 and j < clothResX - 1:
            tri_id = ((i * (clothResX - 1)) + j) * 2
            indices[tri_id * 3+2] = i * clothResX + j
            indices[tri_id * 3+1] = (i + 1) * clothResX + j
            indices[tri_id * 3+0] = i * clothResX + (j + 1)

            tri_id += 1
            indices[tri_id * 3+2] = (i + 1) * clothResX + j + 1
            indices[tri_id * 3+1] = i * clothResX + (j + 1)
            indices[tri_id * 3+0] = (i + 1) * clothResX + j
    ball_centers[0] = ti.Vector([0.0, 3.0, 0.0])
    ball_radius[0]  = 1.0
    deltaT[0] = 0.05

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
        
        
        if (coord_neigh.x >= 0) and (coord_neigh.x <= clothResX) and (coord_neigh.y >= 0) and (coord_neigh.y <= clothResX):
        
            rest_length = get_length2(coord_offcet * ti.Vector([clothWid / clothResX, clothHgt / clothResX]))
            
            p2          = pos[coord_neigh]
            v2          = (p2 - pos_pre[coord_neigh]) / deltaT[0]
            
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
        
    offcet = pos[coord] - ball_centers[0]
    dist   = get_length3(offcet)
    if(dist < ball_radius[0]):  
        delta0         = (ball_radius[0] - dist)
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
            pos[coord]  = pos[coord] * 2.0 - pos_pre[coord] + acc * deltaT[0] * deltaT[0]
            vel[coord]  = (pos[coord] - pos_pre[coord]) /deltaT[0]
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
            pos[coord]  += vel[coord] * deltaT[0]
            vel[coord]  += acc * deltaT[0]
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
            A     = M - deltaT[0] * deltaT[0]* J[coord]
            b     = M@vel[coord] + deltaT[0]*F[coord]
            
            vel[coord] = SolveConjugateGradient(A, ti.Vector([0.0, 0.0, 0.0]), b)
            pos[coord]  += vel[coord] * deltaT[0]
            pos_pre[coord]  = tmp   
            
    
@ti.kernel
def update_verts():
    for i, j in ti.ndrange(clothResX, clothResX):
        vertices[i * clothResX + j] = pos[i, j]
  

gravity    = ti.Vector([0.0, -0.05, 0.0])
ball_centers = ti.Vector.field(3, float, 1)
ball_radius  = ti.field(float, shape=(1))
deltaT       = ti.field(float, shape=(1))

mass       = 1.0
damping    = -0.0125
           
KsStruct   = 50.0
KdStruct   = -0.25
KsShear    = 50.0
KdShear    = -0.25
KsBend     = 50.0
KdBend     = -0.25



gui     = ti.ui.Window('Cloth', (imgSize, imgSize))
canvas = gui.get_canvas()
scene   =  ti.ui.Scene()
camera  = ti.ui.make_camera()
camera.position(5.0, 3.0, 5.0)
camera.lookat(0.0, 3.0, 0.0)
camera.up(0.0, 1.0, 0.0)
mode = 2
reset_cloth()
frame = 0

while gui.running:
    for i in  range(0, 100):
        if mode == 0:
            integrator_explicit()
        if mode == 1:
            integrator_implicit()
        if mode == 2:
            integrator_verlet()

    update_verts()
    scene.mesh(vertices, indices=indices, color=(0.5, 0.5, 0.5))
    scene.particles(ball_centers, radius=0.95, color=(1.0, 0, 0))

    scene.point_light(pos=(10.0, 10.0, 0.0), color=(1.0,1.0,1.0))
    camera.track_user_inputs(gui, movement_speed=0.03, hold_key=ti.ui.LMB)
    scene.set_camera(camera)

    canvas.scene(scene)
    gui.show()
