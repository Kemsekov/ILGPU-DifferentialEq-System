using System.Diagnostics;
using Library;
//solve system of non-linear differential equations

//step size
var dt = 0.01f;

string[] derivatives = [
    "v[0]*v[1]-v[2]*t",        //x'=f_0=x*y-z*t
    "v[0]-v[2]/v[1]",          //y'=f_1=x-z/y
    "v[0]+v[1]+v[2]+f_0(t,v)", //z'=f_2=x+y+z+f_0
];
//x_0=1
//y_0=2
//z_0=3

float[] initialValues = [1,2,3];

//use ILGPU_Derivative.Euler or ILGPU_Derivative.RungeKuttaMethod methods to compute derivatives
var solutions = ILGPU_Derivative.Derivative(derivatives,initialValues,dt,ILGPU_Derivative.Euler);
solutions.First();//initialize 

var timer = new Stopwatch();
timer.Start();
foreach(var s in solutions.Take(10).ToArray()){
    System.Console.WriteLine(string.Join(' ',s));
}
System.Console.WriteLine("Done in "+timer.ElapsedMilliseconds);


