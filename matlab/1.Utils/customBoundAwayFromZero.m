function xBounded=customBoundAwayFromZero(x)

xBounded = x;
xBounded(xBounded < eps('double')) = eps('double');

end