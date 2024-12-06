from scipy import * # contains arrays and special functions
from pylab import * # for plotting

def bessel_upward(l, x):
    """ Upward recursion to compute the spherical bessel function:
         j_{l+1} = (2*l+1)/x * j_l - j_{l-1}
         Works for large x >> l.
         Input:
             l   -- all bessel functions j_l up to l (including l) will be computed
             x   -- j_l(x)
         Output:
             [j_0(x), j_1(x),....j_l(x)]
    """  
    if abs(x)<1e-8:    # Take care of the diverging part first
        j0 = 1.0 - x**2/6.
        j1 = x/3. - x**3/30.
    else:              # The starting values for recursion
        j0 = sin(x)/x
        j1 = j0/x-cos(x)/x
    res=[j0]             # Remember all previous values
    if l==0: return res  # Should work also for l=0 and l=1
    res=[j0,j1]
    if l==1: return res

    # Very small x can create numerical inaccuracy. Taken care of it.
    if abs(j1)<1e-20: return res+zeros(l-1).tolist() 
    
    for i in range(1,l):  # upward recursion
        j2 = j1*(2*i+1)/x - j0
        j0=j1
        j1=j2
        res.append(j2)    # remember j_{l+1} at each step
    return res


def bessel_downward(l, x):
    """ Downward recursion to compute the spherical bessel function:
         j_{l} = (2*l+3)/x * j_{l+1} - j_{l+2}
         Works for small x < l.
         Input:
             l   -- all bessel functions j_l up to l (including l) will be computed
             x   -- j_l(x)
         Output:
             [j_0(x), j_1(x),....j_l(x)]
    """ 
    if (fabs(x)<1e-20): return [1]+zeros(l).tolist() # Small x need to be taken care of
    lstart = l + int(sqrt(40*l)/2.)       # This is a good l to start downward recursion
    j2=0
    j1=1
    res=[]
    for i in range(lstart,-1,-1): # start at lstart, down to 0, including 0
        j0 = (2*i+3)*j1/x-j2      # j_l = (2*l+3)*j_{l+1}/x - j_{l+2}
        j2=j1
        j1=j0
        if i<=l: res.append(j0)  # remember j_l
    true_j0 = sin(x)/x           # correct j_0
    res.reverse()                # reverse the list, becase j_0 was last and j_l was first!
    return array(res)*(true_j0/res[0]) # renormalize!
    

if __name__ == '__main__':
    
    x=arange(1e-5,20.,0.01) # The x-values for bessel evaluation
    n=15
    
    # Exact values of the bessel function
    exact=[]
    for zx in x:
        exact.append( special.sph_jn(n, zx)[0] ) # using scipy function
    exact = array(exact)
    
    # Upward recursive relation for bessel function
    upward=[]
    for zx in x:
        upward.append( bessel_upward(n, zx) )   # upward for all l<=n
    upward = array(upward)
    
    # Upward recursive relation for bessel function
    downward=[]
    for zx in x:
        downward.append( bessel_downward(n, zx) ) # downward for all l<=n
    downward = array(downward)
    
    
    
    for i in range(10,n):  # plotting 10<=i<n
        #plot(x, exact[:,i])
        #plot(x, upward[:,i])
        semilogy(x, abs(upward[:,i]-exact[:,i]), label='n='+str(i)) # log-linear plot
        semilogy(x, abs(downward[:,i]-exact[:,i]), label='n='+str(i)) # adds labels
    
    legend(loc='best')   # puts legend to best location
    xlabel('$x$', fontsize='x-large') # label with large font
    ylabel('Error of bessel functions $j_n(x)$', fontsize='x-large')
    show()               # Only after this call the plots are shown