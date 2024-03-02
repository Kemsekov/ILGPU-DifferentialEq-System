namespace Library;

public class DerivativeMethod
{

    public const string Euler = "newV[i] = prev[i] + dt * f_<i>(t,prev);";
    public const string ImprovedEuler = 
    """
    tmp3=prev[i];
    tmp1=f_<i>(t,prev);
    prev[i]=tmp3 + dt * tmp1;
    newV[i]=tmp3+0.5f*dt*(tmp1+f_<i>(t+dt,prev));
    prev[i]=tmp3;
    """;
    public const string RungeKutta = 
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
}
