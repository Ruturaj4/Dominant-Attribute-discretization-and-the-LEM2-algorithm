! Certain rule set

2, 2
('Wind', 'low') & ('Temperature', 'medium') -> ('Trip', 'yes')
1, 1
('Temperature', 'low') -> ('Trip', 'yes')

! Possible rule set

1, 5
('Wind', 'low') -> ('Trip', 'yes')
1, 3
('Wind', 'medium') -> ('Trip', 'no')
1, 3
('Wind', 'medium') -> ('Trip', 'maybe')
1, 2
('Temperature', 'high') -> ('Trip', 'maybe')
