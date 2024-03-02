using System;

namespace Library
{
    public class DerivativeMethod
    {
        public delegate void DerivMethod(float[] prev,float[] newV,float dt, float t,int i, Func<float,float[],float> f_i);
        public const string Euler = "newV[i] = prev[i] + dt * f_<i>(t,prev);";
        public static void EulerCpu(float[] prev,float[] newV,float dt, float t,int i, Func<float,float[],float> f_i){
            newV[i] = prev[i] + dt * f_i(t,prev);
        }
        public const string ImprovedEuler =
        @"
        tmp3=prev[i];
        tmp1=f_<i>(t,prev);
        prev[i]=tmp3 + dt * tmp1;
        newV[i]=tmp3+0.5f*dt*(tmp1+f_<i>(t+dt,prev));
        prev[i]=tmp3;
        ";
        public static void ImprovedEulerCpu(float[] prev,float[] newV,float dt, float t,int i, Func<float,float[],float> f_i){
            var tmp3=prev[i];
            var tmp1=f_i(t,prev);
            prev[i]=tmp3 + dt * tmp1;
            newV[i]=tmp3+0.5f*dt*(tmp1+f_i(t+dt,prev));
            prev[i]=tmp3;
        }
        public const string RungeKutta =
        @"
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
        ";
        public static void RungeKuttaCpu(float[] prev,float[] newV,float dt, float t,int i, Func<float,float[],float> f_i){
            var tmp1=dt/2;
            var tmp6 = prev[i];
            var tmp2 = f_i(t,prev);
            prev[i]=tmp6+tmp1*tmp2;
            var tmp3 = f_i(t+tmp1,prev);
            prev[i]=tmp6+tmp1*tmp3;
            var tmp4 = f_i(t+tmp1,prev);
            prev[i]=tmp6+dt*tmp4;
            var tmp5 = f_i(t+dt,prev);
            prev[i]=tmp6;
            newV[i] = tmp6+dt/6*(tmp2+2*tmp3+2*tmp4+tmp5);
        }
    }
}