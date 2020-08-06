function xBounded=customBoundAwayFromZero(x)

xBounded = x;
xBounded(xBounded < eps('single')) = eps('single');

end