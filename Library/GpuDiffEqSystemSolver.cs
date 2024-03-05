using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using KernelType = System.Action<ILGPU.Index1D, float, float, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>>;
namespace Library
{

    public class GpuDiffEqSystemSolver : IDisposable, IDiffEqSolver
    {
        private Context context;
        private Accelerator accelerator;
        private int size;
        private Lazy<KernelType> loadedKernel;
        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> </param>
        public GpuDiffEqSystemSolver(string[] derivatives, string derivativeMethod)
        {
            // Initialize ILGPU.
            context = Context.CreateDefault();
            accelerator = context.GetPreferredDevice(preferCPU: false)
                                      .CreateAccelerator(context);
            size = derivatives.Length;
            var derivFunctions =
                derivatives.Select((v, i) => $"float f{i}(float t,ILGPU.ArrayView<float> v)=>{v};")
                .ToArray();

            var derivCases =
                Enumerable.Range(0, size)
                .Select(i => $"case {i}: {derivativeMethod.Replace("<i>", i.ToString())}; break;")
                .ToArray();

            var code =
            @"
            System.Action<ILGPU.Index1D,float,float, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>>
            Execute(ILGPU.Runtime.Accelerator accelerator)
            {
                return ILGPU.Runtime.KernelLoaders
                    .LoadAutoGroupedStreamKernel<ILGPU.Index1D,float,float, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>>
                        (accelerator,Kernel);
            }
            static void Kernel(ILGPU.Index1D i,float t,float dt, ILGPU.ArrayView<float> prev, ILGPU.ArrayView<float> newV){ 

            " + string.Join("\n", derivFunctions) + @"
                //temp values used for computation
                float tmp1 = 0,tmp2 = 0,tmp3 = 0,tmp4 = 0,tmp5 = 0,tmp6 = 0; 
                switch(i){

            " + string.Join("\n", derivCases) + @"

                }
            }
            ";
            loadedKernel = new Lazy<KernelType>(() => DynamicCompilation.CompileFunction<Accelerator, KernelType>(
                code,
                typeof(Index1D), typeof(ArrayView<int>), typeof(KernelLoaders))
            (accelerator));
        }
        /// <summary>
        /// Precompiles kernel
        /// </summary>
        public void CompileKernel()
        {
            var _ = loadedKernel.Value;
        }
        public void Dispose()
        {
            accelerator.Dispose();
            context.Dispose();
        }
        public IEnumerable<(float[] Values, float Time)> EnumerateSolutions(float[] initialValues, float dt, float t0)
        {
            var kernel = loadedKernel.Value;
            //previous values of x,y,z...
            var P = accelerator.Allocate1D<float>(size);
            P.CopyFromCPU(initialValues);
            //new values of x,y,z...
            var V = accelerator.Allocate1D<float>(size);
            
            yield return (P.GetAsArray1D(), t0);

            for (int i = 1; ; i++)
            {
                var t = t0 + i * dt;
                kernel((Index1D)size, t, dt, P.View, V.View);
                accelerator.Synchronize();
                yield return (V.GetAsArray1D(), t);
                (P, V) = (V, P);
            }
        }
    }
}