using System.Diagnostics;
using ILGPU;
using Library;
using ScottPlot;

//solve system of non-linear differential equations

//step size
var dt = 0.01f;

var n = 25;

//time start value
//this means t values going to start from 1 and all initial values
//x_0=x(t0),y_0=y(t0) ... is set on position t=t0
var t0 = 1;

string[] derivatives = [
    "v[0]*v[1]-v[2]*t-System.Math.Sqrt(t)", //x'=f0=xy-zt-sqrt(t)
    "v[0]-v[2]/v[1]*c1",                        //y'=f1=x-z/y
    "v[0]+v[1]+v[2]+f0(t,v)-c0",                //z'=f2=x+y+z+x'
];
string[] constants = ["c0","c1"];
double[] constantsValues = [0,1];

double[] initialValues = [1,2,3];

// use DerivativeMethod.Euler,DerivativeMethod.ImprovedEuler or 
// DerivativeMethod.RungeKutta methods to compute derivatives

using var gpuSolver = new GpuDiffEqSystemSolver(derivatives,DerivativeMethod.ImprovedEuler,constants);
var cpuSolver = new CpuDiffEqSystemSolver(derivatives,DerivativeMethod.ImprovedEulerCpu,constants);

// choose different versions of derivative computation algorithms
var solver = cpuSolver;//gpu solver;

solver.CompileKernel();

var solutionsObj = solver.Solutions(initialValues,dt,t0,constantsValues);
var solutions = solutionsObj.EnumerateSolutions();

solutions.First();//initialize 

var timer = new Stopwatch();
timer.Start();
foreach(var s in solutions.Take(n)){
    var valuesStr = s.Values.Select(t=>t.ToString("0.000"));
    System.Console.WriteLine($"{s.Time:0.000} : {string.Join(' ',valuesStr)}");
}
System.Console.WriteLine("Done in "+timer.ElapsedMilliseconds);


