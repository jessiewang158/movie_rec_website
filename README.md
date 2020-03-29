-In order to run the model 1 no rating:
>from example_norating import make_rec
>make_rec(newmovieId=[2,3],newrating=[3,3])
Note: movieId and rating input needs to be array format

-To run model 2 with rating;
>from example2 import make_rec,similar_movie
>make_rec(newmovieId=[2,3],newrating=[3,3])
>similar_movie(smovieid=[1])
Note: to compute similar movie, the model only take on smovieid as input


-Some Test Results:
1. Model 1 no rating
Test Example 1 using CL2.py:
>result=CF2.recalculate_user([2],[3])
>Output:(3489, 673, 653, 60, 317, 3438, 1702, 2253, 367, 2005)

Test Example 2 using Example.py:
>make_rec (newmovieId=[2,3],newrating=[4,5])
>rec movie (3489, 673, 317, 524, 3438, 653, 2253, 60, 1035, 788)

2. Model 2 with rating
Test Example 1 using CL2.py:
>result=CF2.recalculate_user([2],[3])
>Output:(367, 317, 551, 586, 158, 364, 34, 500, 736, 208)

Test Example 2 using Example.py:
make_rec (newmovieId=[2,3],newrating=[4,5],smovieid=1)
Output: 
>rec movie (788, 62, 653, 736, 317, 104, 5, 7, 141, 1073)
>similarmovie (3114, 170331, 170325, 170339, 170341, 170329, 170335, 170327, 170353)
