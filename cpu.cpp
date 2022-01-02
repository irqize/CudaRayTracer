
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <windows.h>
#include "EasyBMP_1.06/EasyBMP.h"

#define W 1000
#define H W

struct Pixel {
    int r, g, b;
};

struct Material {
    double diffuse, specular_c, specular_k;
};

struct Sphere {
    double pos[3];
    double r;
    double color[3];
    Material mat;
};

struct Light {
    double pos[3];
    double color[3];
    double ambient;
};

struct Camera {
    double pos[3];
    double pt[3];
};

double cpu_dot(double* a, double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double cpu_intersectSphere(double* O, double* ray_dir, Sphere* s) {
    double a = cpu_dot(ray_dir, ray_dir);
    double OS[] = { O[0] - s->pos[0], O[1] - s->pos[1], O[2] - s->pos[2] };
    double b = 2.0 * cpu_dot(ray_dir, OS);
    double c = cpu_dot(OS, OS) - s->r * s->r;
    double disc = b * b - 4 * a * c;

    if (disc > 0.0) {
        double discSqrt = sqrt(disc);
        double q = b < 0.0 ? (-b - discSqrt) / 2.0 : (-b + discSqrt) / 2.0;
        double t0 = q / a;
        double t1 = c / q;
        if (t0 > t1) {
            double temp = t0;
            t0 = t1;
            t1 = temp;
        }
        if (t1 >= 0) {
            if (t0 < 0) {
                return t1;
            }
            else {
                return t0;
            }
        }
    }
    return DBL_MAX / 2.0;
}

double* cpu_normalizeVec3(double* a) {
    double* vec = new double[3];
    double len = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

    vec[0] = a[0] / len;
    vec[1] = a[1] / len;
    vec[2] = a[2] / len;

    return vec;
}

double cpu_clampColor(double x) {
    if (x > 1.0) return 1.0;
    if (x < 0.0) return 0.0;
    return x;
}

void cpu_kernel(Pixel* colors, Sphere* sphere, Light* light, Camera* camera, int idx) {
    int h = idx / W;
    int w = idx % W;

    double pos_x = (2.0 * w) / W - 1.0;
    double pos_y = (2.0 * h) / H - 1.0;

    double _ray_dir[3] = { camera->pt[0] - pos_x, camera->pt[1] - pos_y, camera->pt[2] - camera->pos[2] };
    double* ray_dir = cpu_normalizeVec3(_ray_dir);


    double t = cpu_intersectSphere(camera->pos, ray_dir, sphere);

    if (t > DBL_MAX / 2.0 - 1.0) {
        colors[idx].r = 0;
        colors[idx].g = 0;
        colors[idx].b = 0;
        return;
    }


    double M[] = { camera->pos[0] + ray_dir[0] * t, camera->pos[1] + ray_dir[1] * t, camera->pos[2] + ray_dir[2] * t, };
    double _N[] = { M[0] - sphere->pos[0], M[1] - sphere->pos[1], M[2] - sphere->pos[2] };
    double* N = cpu_normalizeVec3(_N);
    double _toL[] = { light->pos[0] - M[0], light->pos[1] - M[1], light->pos[2] - M[2] };
    double* toL = cpu_normalizeVec3(_toL);
    double _toO[] = { camera->pos[0] - M[0], camera->pos[1] - M[1], camera->pos[2] - M[2] };
    double* toO = cpu_normalizeVec3(_toO);

    double col[] = { light->ambient, light->ambient,light->ambient };
    double _col = sphere->mat.diffuse * (cpu_dot(N, toL) > 0.0 ? cpu_dot(N, toL) : 0.0);

    col[0] += _col * sphere->color[0];
    col[1] += _col * sphere->color[1];
    col[2] += _col * sphere->color[2];


    double _normSum[] = { toL[0] + toO[0], toL[1] + toO[1], toL[2] + toO[2] };
    double* normSum = cpu_normalizeVec3(_normSum);

    double _spM = cpu_dot(N, normSum);
    double spM = pow(_spM > 0.0 ? _spM : 0.0, sphere->mat.specular_k) * sphere->mat.specular_c;

    col[0] += spM * light->color[0];
    col[1] += spM * light->color[1];
    col[2] += spM * light->color[2];

    colors[idx].r = (int)(cpu_clampColor(col[0]) * 255);
    colors[idx].g = (int)(cpu_clampColor(col[1]) * 255);
    colors[idx].b = (int)(cpu_clampColor(col[2]) * 255);


    delete ray_dir;
    delete N;
    delete toL;
    delete toO;
    delete normSum;
}


int main() {
    std::unique_ptr<Pixel[]> colors = std::make_unique<Pixel[]>(H * W);

    BMP Output;
    Output.SetSize(W, H);
    Output.SetBitDepth(24);

    Sphere sphere1 = {
        0.0, 1.0, 1.5,
        1.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 50.0
    };
    Light light1 = {
        5.0, 3.0, -10.0,
        1.0, 1.0, 1.0,
        0.05
    };
    Camera camera = {
        0.0, 1.0, -2.0,
        0.0, 0.0, 0.0
    };

    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    double interval;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
        
    for (int j = 0; j < H * W; j++) {
        cpu_kernel(colors.get(), &sphere1, &light1, &camera, j);
    }

    QueryPerformanceCounter(&end);
    interval = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    printf("[CPU] Time to render the image: %fs\n", interval);


    for (int i = 0; i < W * H; ++i) {
        RGBApixel temp;
        temp.Red = colors[i].r;
        temp.Blue = colors[i].b;
        temp.Green = colors[i].g;
        Output.SetPixel(i % W, i / W, temp);
    }


    Output.WriteToFile("cpu.bmp");

    return 0;
}