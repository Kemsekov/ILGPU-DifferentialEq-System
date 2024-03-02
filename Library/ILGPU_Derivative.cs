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
        tmp1=dt/2;
        tmp6 = prev[i];
        tmp2 = f_<i>(t,prev);
        prev[i]=tmp6+tmp1*tmp2;
        tmp3 = f_<i>(t+tmp1,prev);
        prev[i]=tmp6+tmp1*tmp3;
        tmp4 = f_<i>(t+tmp1,prev);
        prev[i]=tmp6+dt*tmp4;
        tmp5 = f_<i>(t+dt,prev);
        prev[i]=tmp6;
        newV[i] = tmp6+dt/6*(tmp2+2*tmp3+2*tmp4+tmp5);

    """;
    public static IEnumerable<(float[] Values, float Time)> Derivative(string[] derivatives,float[] initialValues,float dt,string methodApply = Euler)
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
            //temp values used for computation
            float tmp1 = 0,tmp2 = 0,tmp3 = 0,tmp4 = 0,tmp5 = 0,tmp6 = 0; 
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
            yield return (V.GetAsArray1D(),t);
            (P, V) = (V, P);
        }
    }
}
