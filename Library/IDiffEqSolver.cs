using System;
using System.Collections.Generic;

namespace Library
{
    public interface IDiffEqSolver
    {
        /// <summary>
        /// Precompiles kernel
        /// </summary>
        void CompileKernel();
        IEnumerable<(double[] Values, double Time)> EnumerateSolutions(double[] initialValues, double dt, double t0);
    }
}
