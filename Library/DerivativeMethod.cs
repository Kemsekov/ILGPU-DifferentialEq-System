using System;

namespace Library
{
    public class DerivativeMethod
    {
        public delegate void DerivMethod(float[] prev,float[] newV,float dt, float t,int i, Func<float,float[],float> f_i);
        public const string Euler = "newV[i] = prev[i] + dt * f<i>(t,prev);";
        public static void EulerCpu(float[] prev,float[] newV,float dt, float t,int i, Func<float,float[],float> f_i){
            newV[i] = prev[i] + dt * f_i(t,prev);
        }
        public const string ImprovedEuler =
        @"
        tmp3=prev[i];
        tmp1=f<i>(t,prev);
        prev[i]=tmp3 + dt * tmp1;
        newV[i]=tmp3+0.5f*dt*(tmp1+f<i>(t+dt,prev));
        prev[i]=tmp3;
        ";
        public static void ImprovedEulerCpu(float[] prev,float[] newV,float dt, float t,int i, Func<float,float[],float> f_i){
            var originalPrev=prev[i];
            var deriv=f_i(t,prev);
            prev[i]=originalPrev + dt * deriv;
            newV[i]=originalPrev+0.5f*dt*(deriv+f_i(t+dt,prev));
            prev[i]=originalPrev;
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
        public static void RungeKuttaCpu(float[] prev,float[] newV,float dt, float t,int i, Func<float,float[],float> f_i){
            var dtHalf=dt/2;
            var originalPrev = prev[i];
            var k1 = f_i(t,prev);
            prev[i]=originalPrev+dtHalf*k1;
            var k2 = f_i(t+dtHalf,prev);
            prev[i]=originalPrev+dtHalf*k2;
            var k3 = f_i(t+dtHalf,prev);
            prev[i]=originalPrev+dt*k3;
            var k4 = f_i(t+dt,prev);
            prev[i]=originalPrev;
            newV[i] = originalPrev+dt/6*(k1+2*k2+2*k3+k4);
        }
    }
}