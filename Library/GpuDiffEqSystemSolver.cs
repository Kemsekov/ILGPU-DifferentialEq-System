using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using KernelType = System.Action<ILGPU.Index1D, double, double, ILGPU.ArrayView<double>, ILGPU.ArrayView<double>>;

namespace Library
{
    public class GpuDiffEqSystemSolver : IDisposable
    {
        Dictionary<string, int> _constantNameToId;
        Context context;
        Accelerator accelerator;
        private int size;
        private Lazy<KernelType> loadedKernel;
        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="constants">A list of constants names used in derivative definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> </param>
        public GpuDiffEqSystemSolver(string[] derivatives, string derivativeMethod,string[]? constants = null)
        {
            constants ??= new string[]{};
            _constantNameToId = constants.Select((i,v)=>(i,v)).ToDictionary(v=>v.i,v=>v.v);
            // Initialize ILGPU.
            context = Context.CreateDefault();
            accelerator = context.GetPreferredDevice(preferCPU: false)
                                      .CreateAccelerator(context);
            size = derivatives.Length;
            derivatives=derivatives.Select(d=>{
                foreach(var c in _constantNameToId){
                    d=d.Replace(c.Key,$"v[{size+c.Value}]");
                }
                d = d.Replace("Math", "System.Math");
                return d;
            }).ToArray();
            var derivFunctions =
                derivatives.Select((v, i) => $"double f{i}(double t,ILGPU.ArrayView<double> v)=>{v};")
                .ToArray();

            var derivCases =
                Enumerable.Range(0, size)
                .Select(i => $"case {i}: {derivativeMethod.Replace("<i>", i.ToString())}; break;")
                .ToArray();

            var code =
            @"
            System.Action<ILGPU.Index1D,double,double, ILGPU.ArrayView<double>, ILGPU.ArrayView<double>>
            Execute(ILGPU.Runtime.Accelerator accelerator)
            {
                return ILGPU.Runtime.KernelLoaders
                    .LoadAutoGroupedStreamKernel<ILGPU.Index1D,double,double, ILGPU.ArrayView<double>, ILGPU.ArrayView<double>>
                        (accelerator,Kernel);
            }
            static void Kernel(ILGPU.Index1D i,double t,double dt, ILGPU.ArrayView<double> prev, ILGPU.ArrayView<double> newV){ 

            " + string.Join("\n", derivFunctions) + @"
                //temp values used for computation
                double tmp1 = 0,tmp2 = 0,tmp3 = 0,tmp4 = 0,tmp5 = 0,tmp6 = 0; 
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

        public SolutionsGpu Solutions(double[] initialValues, double dt, double t0,double[]? constants = null)
        {
            var kernel = loadedKernel.Value;
            //previous values of x,y,z...
            var P = accelerator.Allocate1D<double>(size+_constantNameToId.Count);
            P.View.SubView(0,size).CopyFromCPU(initialValues);
            //new values of x,y,z...
            var V = accelerator.Allocate1D<double>(size+_constantNameToId.Count);
            
            return new SolutionsGpu(size,_constantNameToId.Count,accelerator,P,V,kernel,dt,t0,constants);
        }
    }
}