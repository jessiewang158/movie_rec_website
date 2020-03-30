-In order to run the model 1 no rating:
>from example_norating import make_rec
>make_rec(newmovieId=[2,3],newrating=[3,3])
Note: movieId and rating input needs to be array format

-To run model 2 with rating;
>from example2 import make_rec,similar_movie
>make_rec(newmovieId=[2,3],newrating=[3,3])
>similar_movie(smovieid=[1])
Note: to compute similar movie, the model only take on smovieid as input

-To run model 3 with less training data;
>from example_lessdata import make_rec
>make_rec(newmovieId=[2,3],newrating=[3,3])
>similar_movie(smovieid=[1])
Note: to compute similar movie, the model only take on smovieid as input

-Evaluation:
1.Model 1:
>0.304724863256259 0.16843606209022652
2.Model 2:
>0.5583365843259134 0.4397960842268497
3.Model 3:
>0.3052618304577666 0.16901032561907678

-Some Test Results:
1. Model 1 no rating
Test Example 1 using CL2.py:
>result=CF2.recalculate_user([2],[3])
>Output: (586, 317, 788, 367, 19, 500, 551, 10, 597, 3489)


Test Example 2 using Example.py:
>make_rec (newmovieId=[2,3],newrating=[4,5])
>recmovie (788, 5, 7, 653, 104, 317, 141, 708, 786, 736)


2. Model 2 with rating
Test Example 1 using CL2.py:
>result=CF2.recalculate_user([2],[3])
>Output:(367, 317, 551, 586, 158, 364, 34, 500, 736, 208)

Test Example 2 using Example.py:
make_rec (newmovieId=[2,3],newrating=[4,5],smovieid=1)
Output: 
>rec movie (788, 62, 653, 736, 317, 104, 5, 7, 141, 1073)
>similarmovie (3114, 170331, 170325, 170339, 170341, 170329, 170335, 170327, 170353)

3. Model 3 with less training data
Test Example 1 using CL2.py:
>result=CF2.recalculate_user([2],[3])
>Output:(3489, 673, 653, 60, 317, 3438, 1702, 2253, 367, 2005)

Test Example 2 using Example.py:
>make_rec (newmovieId=[2,3],newrating=[4,5])
>rec movie (3489, 673, 317, 524, 3438, 653, 2253, 60, 1035, 788)
