
A = [ 1 1 1;
      2 2 2;
      3 3 3 ];
  
divisor= [1; 2; 3];

s= std(A)
m = mean(A)

A_norm = (A - m)./s
