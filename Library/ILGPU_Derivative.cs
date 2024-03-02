using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using KernelType = System.Action<ILGPU.Index1D,float, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>>;
namespace Library;

public class ILGPU_Derivative
{

    public const string Euler = "newV[i] = prev[i] + dt * f_<i>(t,prev);";
    public const string RungeKuttaMethod = 
    """
        var halfDt<i>=dt/2;
        var prevOriginal<i> = prev[i];
        var k1<i> = f_<i>(t,prev);
        prev[i]=prevOriginal<i>+halfDt<i>*k1<i>;
        var k2<i> = f_<i>(t+halfDt<i>,prev);
        prev[i]=prevOriginal<i>+halfDt<i>*k2<i>;
        var k3<i> = f_<i>(t+halfDt<i>,prev);
        prev[i]=prevOriginal<i>+dt*k3<i>;
        var k4<i> = f_<i>(t+dt,prev);
        prev[i]=prevOriginal<i>;
        newV[i] = prevOriginal<i>+dt/6*(k1<i>+2*k2<i>+2*k3<i>+k4<i>);

    """;
    public static IEnumerable<float[]> Derivative(string[] derivatives,float[] initialValues,float dt,string methodApply = Euler)
    {
        // Initialize ILGPU.
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(preferCPU: false)
                                  .CreateAccelerator(context);

        var size = derivatives.Length;

        var derivFunctions =
            derivatives.Select((v, i) => $"float f_{i}(float t,ILGPU.ArrayView<float> v)=>{v};")
            .ToArray();

        var derivCases =
            Enumerable.Range(0, size)
            .Select(i => $"case {i}: {methodApply.Replace("<i>",i.ToString())}; break;")
            .ToArray();

        var code =
        """
        System.Action<ILGPU.Index1D,float, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>>
        Execute(ILGPU.Runtime.Accelerator accelerator)
        {
            return ILGPU.Runtime.KernelLoaders
                .LoadAutoGroupedStreamKernel<ILGPU.Index1D,float, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>>
                    (accelerator,Kernel);
        }
        static void Kernel(ILGPU.Index1D i,float t, ILGPU.ArrayView<float> prev, ILGPU.ArrayView<float> newV){ 
        """ + $"var dt = {dt}f;" + """ 

        """ +  string.Join("\n", derivFunctions) + """

            switch(i){

        """ + string.Join("\n", derivCases) + """

            }
        }
        """;
        var loadedKernel = DynamicCompilation.CompileFunction<Accelerator, KernelType>(
            code,
            typeof(Index1D), typeof(ArrayView<int>), typeof(KernelLoaders))
        (accelerator);
        //previous values of x,y,z
        var P = accelerator.Allocate1D<float>(size);
        P.CopyFromCPU(initialValues);
        //new values of x,y,z
        var V = accelerator.Allocate1D<float>(size);

        for(int i = 0;;i++)
        {
            var t = i*dt;
            loadedKernel((Index1D)size,t, P.View, V.View);
            accelerator.Synchronize();
            yield return V.GetAsArray1D();;
            (P, V) = (V, P);
        }
    }
}
