using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using static Library.DerivativeMethod;
namespace Library
{
    public class CpuDiffEqSystemSolver : IDiffEqSolver
    {
        private int size;
        private DerivMethod derivativeMethod;
        private Lazy<Func<float, float[], float>[]> functions;

        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> cpu </param>
        public CpuDiffEqSystemSolver(Func<float, float[], float>[] derivatives, DerivMethod derivativeMethod){
            size = derivatives.Length;
            functions = new Lazy<Func<float,float[],float>[]>(() => derivatives);
            this.derivativeMethod=derivativeMethod;
        }
        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> cpu </param>
        public CpuDiffEqSystemSolver(string[] derivatives, DerivMethod derivativeMethod)
        {
            size = derivatives.Length;
            var derivFunctions =
                derivatives.Select((v, i) => $"float f_{i}(float t,float[] v)=>{v};")
                .ToArray();
            var funcDecl = derivatives.Select((v, i)=>$"f_{i}").ToArray();
            var code =
            @"
            System.Func<float,float[],float>[]
            Execute(int a)
            {
                " + string.Join("\n", derivFunctions) + @"
                return "+$"new System.Func<float,float[],float>[]{{ {string.Join(",",funcDecl)} }} ;"+@"
            }
            ";
            this.derivativeMethod=derivativeMethod;
            functions = new Lazy<Func<float,float[],float>[]>(() => DynamicCompilation.CompileFunction<int, Func<float,float[],float>[]>(code)(1));
        }
        /// <summary>
        /// Precompiles kernel
        /// </summary>
        public void CompileKernel()
        {
            var _ = functions.Value;
        }
        public IEnumerable<(float[] Values, float Time)> EnumerateSolutions(float[] initialValues, float dt, float t0)
        {
            //previous values of x,y,z...
            var P = initialValues.ToArray();
            //new values of x,y,z...
            var V = new float[size];

            for (int i = 0; ; i++)
            {
                var t = t0 + i * dt;
                _Kernel(t,P,V,dt);
                yield return (V, t);
                (P, V) = (V, P);
            }
        }

        private void _Kernel(float t, float[] p, float[] v, float dt)
        {
            var funcs = functions.Value;
            Parallel.For(0,size,i=>derivativeMethod(p,v,dt,t,i,funcs[i]));
        }
    }
}