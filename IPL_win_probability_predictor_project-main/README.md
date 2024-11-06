# IPL_win_probability_predictor_project
In this project I develop the model which predict which team will win the IPL match on some features. Here are the 17 features on which i make a project.

features:-

1. ID                 Id of player  
2. Season             IPL season 2017  
3. city               City ipl played
4. date               On which date which team play    
5. team1              Team 1
6. team2              Team 2   
7. toss_winner        Who win Toss
8. toss_decision      After win toss what team selectd   
9. result             Result  
10. winner            who win match
11. win_by_runs       win by runs a team      
12. win_by_wickets    win by wickets    
13. player_of_match   Man of the match       
14. venue             Where it play
15. umpire1           who is 1st umpire
16. umpire2           who is 2st umpire
17. umpire3           who is 3st umpire

Libraries Used in project :-

Pandas
Numpy
Scikit-learn
Matplot lib
joblib

Import .csv file in jupyter notebook and first perform some Pandas library function on that data to maka perfect. we can get deep details about our data using .info() function.

With the help of numpy i got the mathimatical details of data which is helpfull further. i got count, min, max, standard deviation etc details of data. i used random function to arrange data randomly . i make the different histograms for analyse the data.

Then i split the data into train set and test set with the help of sklearn.model_selection. i use train_test_split for spliting data. Now check the correlation of functions and make the plots of some relevent features. According plots trying out features combinations.
