using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Threading;
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
        private Lazy<Func<double, double[], double>[]> functions;

        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> cpu </param>
        public CpuDiffEqSystemSolver(Func<double, double[], double>[] derivatives, DerivMethod derivativeMethod)
        {
            size = derivatives.Length;
            functions = new Lazy<Func<double, double[], double>[]>(() => derivatives);
            this.derivativeMethod = derivativeMethod;
        }
        /// <param name="derivatives">A list of derivatives definitions</param>
        /// <param name="derivativeMethod">Some of <see cref="DerivativeMethod"/> cpu </param>
        public CpuDiffEqSystemSolver(string[] derivatives, DerivMethod derivativeMethod)
        {
            size = derivatives.Length;
            var derivFunctions =
                derivatives.Select((v, i) => $"double f{i}(double t,double[] v)=>{v};")
                .ToArray();
            var funcDecl = derivatives.Select((v, i) => $"f{i}").ToArray();
            var code =
            @"
            System.Func<double,double[],double>[]
            Execute(int a)
            {
                " + string.Join("\n", derivFunctions) + @"
                return " + $"new System.Func<double,double[],double>[]{{ {string.Join(",", funcDecl)} }} ;" + @"
            }
            ";
            this.derivativeMethod = derivativeMethod;
            functions = new Lazy<Func<double, double[], double>[]>(() => DynamicCompilation.CompileFunction<int, Func<double, double[], double>[]>(code)(1));
        }
        /// <summary>
        /// Precompiles kernel
        /// </summary>
        public void CompileKernel()
        {
            var _ = functions.Value;
        }
        public IEnumerable<(double[] Values, double Time)> EnumerateSolutions(double[] initialValues, double dt, double t0)
        {
            //previous values of x,y,z...
            var P = initialValues.ToArray();
            //new values of x,y,z...
            var V = new double[size];

            yield return (P, t0);

            for (int i = 1; ; i++)
            {
                var t = t0 + i * dt;
                _Kernel(t, P, V, dt);
                yield return (V, t);
                (P, V) = (V, P);
            }
        }
        private void _Kernel(double t, double[] p, double[] v, double dt)
        {
            var funcs = functions.Value;
            Parallel.For(0, size, i => v[i]=derivativeMethod(p, dt, t, i, funcs[i]));
        }
       
    }
}