using System.Diagnostics;
using ILGPU;
using Library;
using ScottPlot;
using Array3DView = ILGPU.Runtime.ArrayView3D<double, ILGPU.Stride3D.DenseXY>;
using Array3D = ILGPU.Runtime.MemoryBuffer3D<double, ILGPU.Stride3D.DenseXY>;
using Microsoft.CodeAnalysis;
using ILGPU.Runtime;
void Move(double[][,,] buffer, Array3DView[] P)
{
    for (int i = 0; i < P.Length; i++)
    {
        P[i].CopyToCPU(buffer[i]);
    }
}
//solve system of non-linear differential equations

//step size
var dt = 0.001;

var n = 1000;

//time start value
//this means t values going to start from 1 and all initial values
//x_0=x(t0),y_0=y(t0) ... is set on position t=t0
var t0 = 1;

string[] derivatives = [
    "c0*(v0_xx+v0_yy)+c1", //x'=f0=xy-zt-sqrt(t)
];
string[] partial = ["v0_xx", "v0_yy"];
string[] constants = ["c0","c1"];
double[] constantsValues = [4.1,0];
//grid step size
var h = 0.1;
var xsize = 100;
var ysize = 100;

double[][,,] initialValues = [new double[xsize, ysize, 1]];

foreach (var init in initialValues)
{
    for (int i = 0; i < xsize; i++)
        for (int j = 0; j < ysize; j++)
            init[i, j, 0] = Random.Shared.NextDouble();
}
foreach (var init in initialValues)
{
    for (int i = 0; i < 20; i++)
        for (int j = 0; j < 30; j++)
        {
            init[i + 20, j + 50, 0] = Random.Shared.NextDouble()*30; ;
        }
}


// use DerivativeMethod.Euler,DerivativeMethod.ImprovedEuler or 
// DerivativeMethod.RungeKutta methods to compute derivatives

// using var gpuSolver = new GpuDiffEqSystemSolver(derivatives,DerivativeMethod.ImprovedEuler);
var gpuSolver = new GpuDiffEqSystemSolver3D(derivatives, DerivativeMethod.RungeKutta, partial,constants);
var cpuSolver = new CpuDiffEqSystemSolver3D(derivatives, DerivativeMethod.RungeKuttaCpu, partial,constants);

var solver = gpuSolver;
// choose different versions of derivative computation algorithms
// IDiffEqSolver solver = cpuSolver;//gpu solver;

solver.CompileKernel();

var solutionsObj = solver.Solutions(initialValues, dt, t0, h, 1, 2, 3,constantsValues);
var solutions = solutionsObj.EnumerateSolutions();
solutions.First();//initialize 

var timer = new Stopwatch();
timer.Start();
var counter = 0;
foreach (var s in solutions.Take(n))
{
    counter++;
    if (counter % 100 == 0)
    {
        var data = s.Values[0].ToCpu();
        var plt = new Plot();
        var res = new double[xsize, ysize];
        
        Buffer.BlockCopy(data, 0, res, 0, xsize * ysize * sizeof(double));
        plt.Add.Heatmap(res);
        plt.SavePng($"images/{counter/100}.png", 500, 500);
    }
    System.Console.WriteLine($"{s.Time:0.000}");
}
System.Console.WriteLine("Done in " + timer.ElapsedMilliseconds);


