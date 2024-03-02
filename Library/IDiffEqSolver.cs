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
        IEnumerable<(float[] Values, float Time)> EnumerateSolutions(float[] initialValues, float dt, float t0);
    }
}
