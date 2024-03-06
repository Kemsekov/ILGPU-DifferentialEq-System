using System;

namespace Library
{
    public class DerivativeMethod
    {
        public delegate void DerivMethod(float[] prev, float[] newV, float dt, float t, int i, Func<float, float[], float> f_i);
        public const string Euler = "newV[i] = prev[i] + dt * f<i>(t,prev);";
        public static void EulerCpu(float[] prev, float[] newV, float dt, float t, int i, Func<float, float[], float> f_i)
        {
            newV[i] = prev[i] + dt * f_i(t, prev);
        }
        public const string ImprovedEuler =
        @"
        tmp3=prev[i];
        tmp1=f<i>(t,prev);
        prev[i]=tmp3 + dt * tmp1;
        newV[i]=tmp3+0.5f*dt*(tmp1+f<i>(t+dt,prev));
        prev[i]=tmp3;
        ";
        public static void ImprovedEulerCpu(float[] prev, float[] newV, float dt, float t, int i, Func<float, float[], float> f_i)
        {
            var originalPrev = prev[i];
            var deriv = f_i(t, prev);
            prev[i] = originalPrev + dt * deriv;
            newV[i] = originalPrev + 0.5f * dt * (deriv + f_i(t + dt, prev));
            prev[i] = originalPrev;
        }
        public const string RungeKutta =
        @"
        tmp1=dt/2;
        tmp6 = prev[i];
        tmp2 = f<i>(t,prev);
        prev[i]=tmp6+tmp1*tmp2;
        tmp3 = f<i>(t+tmp1,prev);
        prev[i]=tmp6+tmp1*tmp3;
        tmp4 = f<i>(t+tmp1,prev);
        prev[i]=tmp6+dt*tmp4;
        tmp5 = f<i>(t+dt,prev);
        prev[i]=tmp6;
        newV[i] = tmp6+dt/6*(tmp2+2*tmp3+2*tmp4+tmp5);
        ";
        public static void RungeKuttaCpu(float[] prev, float[] newV, float dt, float t, int i, Func<float, float[], float> f_i)
        {
            var dtHalf = dt / 2;
            var originalPrev = prev[i];
            var k1 = f_i(t, prev);
            prev[i] = originalPrev + dtHalf * k1;
            var k2 = f_i(t + dtHalf, prev);
            prev[i] = originalPrev + dtHalf * k2;
            var k3 = f_i(t + dtHalf, prev);
            prev[i] = originalPrev + dt * k3;
            var k4 = f_i(t + dt, prev);
            prev[i] = originalPrev;
            newV[i] = originalPrev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
        /// <summary>
        /// Approximates first derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="by">Dimension to take derivative of</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public float Derivative(int i, int j, int k, int by, float h, float[,,] grid)
        {
            switch (by)
            {
                case 0:
                    return DerivativeX(i, j, k, h, grid);
                case 1:
                    return DerivativeY(i, j, k, h, grid);
                case 2:
                    return DerivativeZ(i, j, k, h, grid);
            }
            throw new ArgumentException("Cannot find derivative at index " + by + " for 3-dimensional tensor");
        }

        /// <summary>
        /// Approximates first z derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public static float DerivativeZ(int i, int j, int k, float h, float[,,] grid)
        {
            int size2 = grid.GetLength(2);
            if (k > 1 && k < size2 - 2)
                return CentralDifference4Order(
                    h,
                    grid[i, j, k + 1],
                    grid[i, j, k + 2],
                    grid[i, j, k - 1],
                    grid[i, j, k - 2]
                );
            if (k > 0 && k < size2 - 1)
                return CentralDifference(
                    h,
                    grid[i, j, k + 1],
                    grid[i, j, k - 1]
                );
            if (k == 0)
                return ForwardDifference3Points(
                    h,
                    grid[i, j, k],
                    grid[i, j, k + 1],
                    grid[i, j, k + 2]
                );
            return BackwardDifference3Points(
                h,
                grid[i, j, k],
                grid[i, j, k - 1],
                grid[i, j, k - 2]
            );
        }

        /// <summary>
        /// Approximates first y derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public static float DerivativeY(int i, int j, int k, float h, float[,,] grid)
        {
            int size1 = grid.GetLength(1);
            if (j > 1 && j < size1 - 2)
                return CentralDifference4Order(
                    h,
                    grid[i, j + 1, k],
                    grid[i, j + 2, k],
                    grid[i, j - 1, k],
                    grid[i, j - 2, k]
                );
            if (j > 0 && j < size1 - 1)
                return CentralDifference(
                    h,
                    grid[i, j + 1, k],
                    grid[i, j - 1, k]
                );
            if (j == 0)
                return ForwardDifference3Points(
                    h,
                    grid[i, j, k],
                    grid[i, j + 1, k],
                    grid[i, j + 2, k]
                );
            return BackwardDifference3Points(
                h,
                grid[i, j, k],
                grid[i, j - 1, k],
                grid[i, j - 2, k]
            );
        }

        /// <summary>
        /// Approximates first x derivative of tensor-defined function at point (<paramref name="i"/>,<paramref name="j"/>,<paramref name="k"/>) on <paramref name="by"/> dimension
        /// </summary>
        /// <param name="h">discretization size</param>
        /// <param name="i">Index 0</param>
        /// <param name="j">Index 1</param>
        /// <param name="k">Index 2</param>
        /// <param name="grid"></param>
        /// <returns>derivative approximation</returns>
        public static float DerivativeX(int i, int j, int k, float h, float[,,] grid)
        {
            var size0 = grid.GetLength(0);
            if (i > 1 && i < size0 - 2)
                return CentralDifference4Order(
                    h,
                    grid[i + 1, j, k],
                    grid[i + 2, j, k],
                    grid[i - 1, j, k],
                    grid[i - 2, j, k]
                );
            if (i > 0 && i < size0 - 1)
                return CentralDifference(
                    h,
                    grid[i + 1, j, k],
                    grid[i - 1, j, k]
                );
            if (i == 0)
                return ForwardDifference3Points(
                    h,
                    grid[i, j, k],
                    grid[i + 1, j, k],
                    grid[i + 2, j, k]
                );
            return BackwardDifference3Points(
                h,
                grid[i, j, k],
                grid[i - 1, j, k],
                grid[i - 2, j, k]
            );
        }

        static float BackwardDifference3Points(float h, float u, float u_m, float u_mm)
        {
            return (3 * u - 4 * u_m + u_mm) * 0.5f / h;
        }
        static float ForwardDifference3Points(float h, float u, float u_p, float u_pp)
        {
            return (-3 * u + 4 * u_p - u_pp) * 0.5f / h;
        }
        static float CentralDifference(float h, float u_p, float u_m)
        {
            return (u_p - u_m) * 0.5f / h;
        }
        static float CentralDifference4Order(float h, float u_p, float u_pp, float u_m, float u_mm)
        {
            return 0.0833333333f / h * (8 * (u_p - u_m) + u_mm - u_pp);
        }
    }
}