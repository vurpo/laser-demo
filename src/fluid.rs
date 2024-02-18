use cgmath::Vector3;

const DIFFUSION: f64 = 0.0;
const VISCOSITY: f64 = 0.0;

pub struct FluidSimulation {
    size: (usize, usize, usize),
    scale: f64,
    pub density: Vec<f64>,
    velocity: Vec<Vector3<f64>>,
    density2: Vec<f64>,
    velocity2: Vec<Vector3<f64>>,
    tmp1: Vec<f64>,
    tmp2: Vec<f64>,
}

#[inline]
fn index(x: isize, y: isize, z: isize, size: (usize, usize, usize)) -> usize {
    ((z*size.0 as isize*size.1 as isize)+(y*size.0 as isize)+(x)) as usize
}

impl FluidSimulation {
    pub fn new(size: (usize, usize, usize), scale: f64) -> Self {
        let length = size.0*size.1*size.2;
        return FluidSimulation{
            size,
            scale,
            density: vec![0.; length],
            velocity: vec![Vector3::new(0., 0., 0.); length],
            density2: vec![0.; length],
            velocity2: vec![Vector3::new(0., 0., 0.); length],
            tmp1: vec![0.; length],
            tmp2: vec![0.; length],
        };
    }

    pub fn update(&mut self, dt: f64) {
        let bottom_center = Vector3::new(self.size.0 as f64/2.0, self.size.1 as f64/2.0, 0.0);
        self.density[index(bottom_center.x as isize, bottom_center.y as isize, bottom_center.z as isize, self.size)] += 1.0*dt;
        self.velocity[index(bottom_center.x as isize, bottom_center.y as isize, bottom_center.z as isize, self.size)] = Vector3::new(0.1, 0.0, 1.0);

        let a: f64 = dt * VISCOSITY * self.scale.powf(-2.);
        // diffuse the data from velocity to velocity2
        lin_solve_3(&self.velocity, &mut self.velocity2, a, 1. + 6. * a, 4, self.size);
        
        // project the data in place in velocity2
        project(&mut self.velocity2, &mut self.tmp1, &mut self.tmp2, 4, self.size, self.scale);
        
        // advect the data from velocity2 to velocity
        advect_3(&self.velocity2, &mut self.velocity, dt, self.size, self.scale);
        
        // project the data in place in velocity
        project(&mut self.velocity, &mut self.tmp1, &mut self.tmp2, 4, self.size, self.scale);
        
        let a: f64 = dt * DIFFUSION * self.scale.powf(-2.);
        // diffuse the data from density to density2
        lin_solve(&self.density, &mut self.density2, a, 1. + 6. * a, 4, self.size);

        // advect the data from density2 to density
        advect(&self.density2, &mut self.density, &mut self.velocity, dt, self.size, self.scale);
    }

}
// TODO: generic this

fn lin_solve(old_array: &Vec<f64>, new_array: &mut Vec<f64>, a: f64, c: f64, iter: usize, size: (usize, usize, usize)) {
    for _ in 0..iter {
        for z in 0..size.2 as isize {
            for y in 0..size.1 as isize {
                for x in 0..size.0 as isize {
                    new_array[index(x,y,z,size)] = 
                        (old_array[index(x,y,z,size)]
                        +a*(
                             old_array.get(index(x+1, y   , z   ,size)).unwrap_or(&0.0)
                            +old_array.get(index(x-1, y   , z   ,size)).unwrap_or(&0.0)
                            +old_array.get(index(x  , y+1 , z   ,size)).unwrap_or(&0.0)
                            +old_array.get(index(x  , y-1 , z   ,size)).unwrap_or(&0.0)
                            +old_array.get(index(x  , y   , z+1 ,size)).unwrap_or(&0.0)
                            +old_array.get(index(x  , y   , z-1 ,size)).unwrap_or(&0.0)
                        ))/c;
                }
            }
        }
    }
}

fn lin_solve_3(old_array: &Vec<Vector3<f64>>, new_array: &mut Vec<Vector3<f64>>, a: f64, c: f64, iter: usize, size: (usize, usize, usize)) {
    for _ in 0..iter {
        for z in 0..size.2 as isize {
            for y in 0..size.1 as isize {
                for x in 0..size.0 as isize {
                    new_array[index(x,y,z,size)] = 
                        (old_array[index(x,y,z,size)]
                        +a*(
                             old_array.get(index(x+1, y   , z   ,size)).unwrap_or(&Vector3::new(0.0,0.0,0.0))
                            +old_array.get(index(x-1, y   , z   ,size)).unwrap_or(&Vector3::new(0.0,0.0,0.0))
                            +old_array.get(index(x  , y+1 , z   ,size)).unwrap_or(&Vector3::new(0.0,0.0,0.0))
                            +old_array.get(index(x  , y-1 , z   ,size)).unwrap_or(&Vector3::new(0.0,0.0,0.0))
                            +old_array.get(index(x  , y   , z+1 ,size)).unwrap_or(&Vector3::new(0.0,0.0,0.0))
                            +old_array.get(index(x  , y   , z-1 ,size)).unwrap_or(&Vector3::new(0.0,0.0,0.0))
                        ))/c;
                }
            }
        }
    }
}

fn project(velocity: &mut Vec<Vector3<f64>>, p: &mut Vec<f64>, div: &mut Vec<f64>, iter: usize, size: (usize, usize, usize), scale: f64) {
    for z in 0..size.2 as isize {
        for y in 0..size.1 as isize {
            for x in 0..size.0 as isize {
                div[index(x,y,z,size)] = -0.5*(
                     velocity.get(index(x+1, y   , z   ,size)).map(|v| v.x).unwrap_or(0.0)
                    -velocity.get(index(x-1, y   , z   ,size)).map(|v| v.x).unwrap_or(0.0)
                    +velocity.get(index(x  , y+1 , z   ,size)).map(|v| v.y).unwrap_or(0.0)
                    -velocity.get(index(x  , y-1 , z   ,size)).map(|v| v.y).unwrap_or(0.0)
                    +velocity.get(index(x  , y   , z+1 ,size)).map(|v| v.z).unwrap_or(0.0)
                    -velocity.get(index(x  , y   , z-1 ,size)).map(|v| v.z).unwrap_or(0.0)
                )*scale;
                p[index(x, y, z,size)] = 0.;
            }
        }
    }

    lin_solve(div, p, 1., 6., iter, size);

    for z in 0..size.2 as isize {
        for y in 0..size.1 as isize {
            for x in 0..size.0 as isize {
                velocity[index(x, y, z,size)] -= Vector3::new(
                    0.5 * (p.get(index(x+1, y, z,size)).unwrap_or(&0.0)-p.get(index(x-1, y, z,size)).unwrap_or(&0.0)) / scale,
                    0.5 * (p.get(index(x, y+1, z,size)).unwrap_or(&0.0)-p.get(index(x, y-1, z,size)).unwrap_or(&0.0)) / scale,
                    0.5 * (p.get(index(x, y, z+1,size)).unwrap_or(&0.0)-p.get(index(x, y, z-1,size)).unwrap_or(&0.0)) / scale
                )
            }
        }
    }
}

fn advect_3(old_array: &Vec<Vector3<f64>>, new_array: &mut Vec<Vector3<f64>>, dt: f64, size: (usize, usize, usize), scale: f64) {
    let dt0 = dt*(1.0/scale);
    for iz in 0..size.2 as isize {
        for iy in 0..size.1 as isize {
            for ix in 0..size.0 as isize {
                let tmp1 = dt0 * old_array[index(ix, iy, iz,size)].x;
                let tmp2 = dt0 * old_array[index(ix, iy, iz,size)].y;
                let tmp3 = dt0 * old_array[index(ix, iy, iz,size)].z;
                let mut x = ix as f64 - tmp1;
                let mut y = iy as f64 - tmp2;
                let mut z = iz as f64 - tmp3;

                x = x.clamp(0.5, size.0 as f64+0.5);
                y = y.clamp(0.5, size.1 as f64+0.5);
                z = z.clamp(0.5, size.2 as f64+0.5);

                let x_floor = x.floor();
                let y_floor = y.floor();
                let z_floor = z.floor();

                let x_floor1 = x_floor+1.0;
                let y_floor1 = y_floor+1.0;
                let z_floor1 = z_floor+1.0;

                let x_fract = x.fract();
                let y_fract = y.fract();
                let z_fract = z.fract();

                let x_fract1 = 1.0-x_fract;
                let y_fract1 = 1.0-y_fract;
                let z_fract1 = 1.0-z_fract;

                new_array[index(ix, iy, iz,size)] = Vector3::new(
                    x_fract1 *
                            (y_fract1 * (z_fract1 * old_array.get(index(x_floor as isize, y_floor as isize, z_floor as isize,size)).map(|v| v.x).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor as isize, y_floor as isize, z_floor1 as isize,size)).map(|v| v.x).unwrap_or(0.0))
                            +(y_fract * (z_fract1 * old_array.get(index(x_floor as isize, y_floor1 as isize, z_floor as isize,size)).map(|v| v.x).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor as isize, y_floor1 as isize, z_floor1 as isize,size)).map(|v| v.x).unwrap_or(0.0))))
                    +x_fract *
                            (y_fract1 * (z_fract1 * old_array.get(index(x_floor1 as isize, y_floor as isize, z_floor as isize,size)).map(|v| v.x).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor1 as isize, y_floor as isize, z_floor1 as isize,size)).map(|v| v.x).unwrap_or(0.0))
                            +(y_fract * (z_fract1 * old_array.get(index(x_floor1 as isize, y_floor1 as isize, z_floor as isize,size)).map(|v| v.x).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor1 as isize, y_floor1 as isize, z_floor1 as isize,size)).map(|v| v.x).unwrap_or(0.0)))),
                    x_fract1 *
                            (y_fract1 * (z_fract1 * old_array.get(index(x_floor as isize, y_floor as isize, z_floor as isize,size)).map(|v| v.y).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor as isize, y_floor as isize, z_floor1 as isize,size)).map(|v| v.y).unwrap_or(0.0))
                            +(y_fract * (z_fract1 * old_array.get(index(x_floor as isize, y_floor1 as isize, z_floor as isize,size)).map(|v| v.y).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor as isize, y_floor1 as isize, z_floor1 as isize,size)).map(|v| v.y).unwrap_or(0.0))))
                    +x_fract *
                            (y_fract1 * (z_fract1 * old_array.get(index(x_floor1 as isize, y_floor as isize, z_floor as isize,size)).map(|v| v.y).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor1 as isize, y_floor as isize, z_floor1 as isize,size)).map(|v| v.y).unwrap_or(0.0))
                            +(y_fract * (z_fract1 * old_array.get(index(x_floor1 as isize, y_floor1 as isize, z_floor as isize,size)).map(|v| v.y).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor1 as isize, y_floor1 as isize, z_floor1 as isize,size)).map(|v| v.y).unwrap_or(0.0)))),
                    x_fract1 *
                            (y_fract1 * (z_fract1 * old_array.get(index(x_floor as isize, y_floor as isize, z_floor as isize,size)).map(|v| v.z).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor as isize, y_floor as isize, z_floor1 as isize,size)).map(|v| v.z).unwrap_or(0.0))
                            +(y_fract * (z_fract1 * old_array.get(index(x_floor as isize, y_floor1 as isize, z_floor as isize,size)).map(|v| v.z).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor as isize, y_floor1 as isize, z_floor1 as isize,size)).map(|v| v.z).unwrap_or(0.0))))
                    +x_fract *
                            (y_fract1 * (z_fract1 * old_array.get(index(x_floor1 as isize, y_floor as isize, z_floor as isize,size)).map(|v| v.z).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor1 as isize, y_floor as isize, z_floor1 as isize,size)).map(|v| v.z).unwrap_or(0.0))
                            +(y_fract * (z_fract1 * old_array.get(index(x_floor1 as isize, y_floor1 as isize, z_floor as isize,size)).map(|v| v.z).unwrap_or(0.0)
                                    +z_fract * old_array.get(index(x_floor1 as isize, y_floor1 as isize, z_floor1 as isize,size)).map(|v| v.z).unwrap_or(0.0)))),
                );
            }
        }
    }
}

fn advect(old_array: &Vec<f64>, new_array: &mut Vec<f64>, velocity: &mut Vec<Vector3<f64>>, dt: f64, size: (usize, usize, usize), scale: f64) {
    let dt0 = dt*(1.0/scale);
    for iz in 0..size.2 as isize {
        for iy in 0..size.1 as isize {
            for ix in 0..size.0 as isize {
                let tmp1 = dt0 * velocity[index(ix, iy, iz,size)].x;
                let tmp2 = dt0 * velocity[index(ix, iy, iz,size)].y;
                let tmp3 = dt0 * velocity[index(ix, iy, iz,size)].z;
                let mut x = ix as f64 - tmp1;
                let mut y = iy as f64 - tmp2;
                let mut z = iz as f64 - tmp3;

                x = x.clamp(0.5, size.0 as f64+0.5);
                y = y.clamp(0.5, size.1 as f64+0.5);
                z = z.clamp(0.5, size.2 as f64+0.5);

                let x0 = x.floor();
                let y0 = y.floor();
                let z0 = z.floor();

                let x1 = x0+1.0;
                let y1 = y0+1.0;
                let z1 = z0+1.0;

                let x2 = x.fract();
                let y2 = y.fract();
                let z2 = z.fract();

                let x3 = 1.0-x2;
                let y3 = 1.0-y2;
                let z3 = 1.0-z2;

                new_array[index(ix, iy, iz,size)] =
                    x3 *
                            (y3 * (z3 * old_array.get(index(x0 as isize, y0 as isize, z0 as isize,size)).unwrap_or(&0.0)
                                    +z2 * old_array.get(index(x0 as isize, y0 as isize, z1 as isize,size)).unwrap_or(&0.0))
                            +(y2 * (z3 * old_array.get(index(x0 as isize, y1 as isize, z0 as isize,size)).unwrap_or(&0.0)
                                    +z2 * old_array.get(index(x0 as isize, y1 as isize, z1 as isize,size)).unwrap_or(&0.0))))
                    +x2 *
                            (y3 * (z3 * old_array.get(index(x1 as isize, y0 as isize, z0 as isize,size)).unwrap_or(&0.0)
                                    +z2 * old_array.get(index(x1 as isize, y0 as isize, z1 as isize,size)).unwrap_or(&0.0))
                            +(y2 * (z3 * old_array.get(index(x1 as isize, y1 as isize, z0 as isize,size)).unwrap_or(&0.0)
                                    +z2 * old_array.get(index(x1 as isize, y1 as isize, z1 as isize,size)).unwrap_or(&0.0))))
                ;
            }
        }
    }
}