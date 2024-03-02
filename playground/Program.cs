using System.Diagnostics;
using Library;
//solve system of non-linear differential equations

//step size
var dt = 0.01f;

//time start value
//this means t values going to start from 1 and all initial values
//x_0=x(t0),y_0=y(t0) ... is set on position t=t0
var t0 = 1;

string[] derivatives = [
    "v[0]*v[1]-v[2]*t-System.MathF.Sqrt(t)", //x'=f_0=xy-zt-sqrt(t)
    "v[0]-v[2]/v[1]",                        //y'=f_1=x-z/y
    "v[0]+v[1]+v[2]+f_0(t,v)",               //z'=f_2=x+y+z+x'
];

float[] initialValues = [1,2,3];


// use DerivativeMethod.Euler,DerivativeMethod.ImprovedEuler or 
// DerivativeMethod.RungeKutta methods to compute derivatives
using var solver = new GpuDiffEqSystemSolver(derivatives,DerivativeMethod.RungeKutta);
// var solver = new CpuDiffEqSystemSolver(derivatives,DerivativeMethod.RungeKuttaCpu);

solver.CompileKernel();

var solutions = solver.EnumerateSolutions(initialValues,dt,t0);
solutions.First();//initialize 

var timer = new Stopwatch();
timer.Start();
foreach(var s in solutions.Take(20)){
    var valuesStr = s.Values.Select(t=>t.ToString("0.000"));
    System.Console.WriteLine($"{s.Time:0.000} : {string.Join(' ',valuesStr)}");
}
System.Console.WriteLine("Done in "+timer.ElapsedMilliseconds);


