# Purpose: This code produces 8 interactions from 18 univariate significant QTL determining BMI in rats
# Date: 7/21/2022
# Description:
  # The phenotype is ranked and ordered from least to greatest
  # A random proportion of the data is selected to produce ranked interaction terms
    # Increasing the proportion of the data that is ranked increases the strength of the interactions
    # The leftover proportion of the data is randomly shuffled
  # For each interaction, two features are shuffled, and an XOR interaction term is calclulated
  # The two features are ranked by the value of their XOR interaction term (0 -> 1) and then re-combined with the ranked phenotype
  # Features that do not participate in an interaction are randomly shuffled to break their univariate effects
  # These components are combined column-wise to produce the final dataset
  # Running these datasets through autoQTL should produce a top testing R^2 between around 0 to 0.2, depending on the number of interactions set
  # Areas are commented out where modifications can be made to reduce interactions down incrementally to 0
    #---THESE AREAS ARE COMMENTED OUT WITH THIS STYLE---#

#tidyverse is used for operations on dataframes
library(tidyverse)

#set seed
set.seed(24)

############prepare data############

#read in data
BMIwTail <- read.csv("BMIwTail.csv") ##filename

#add in a column y_rank that ranks phenotype values from least to greatest
BMIwTail <- BMIwTail %>% 
  mutate(y_rank = rank(bmi_w_tail_res_res))

#create a dataframe ordered by phenotype rank
y_rankedBMIwTail <- BMIwTail[order(BMIwTail$y_rank), ] 

#create a dataframe with all features shuffled
#features that do not participate in an interaction are taken from this datafrmae when making final data later
y_rankedBMIwTail_shuff <- y_rankedBMIwTail %>% 
  mutate(chr5.107167969_G = sample(chr5.107167969_G), 
         chr9.71715296_A = sample(chr9.71715296_A)) %>% 
  mutate(chr6.29889998_C = sample(chr6.29889998_C), 
         chr7.129118847_C = sample(chr7.129118847_C)) %>% 
  mutate(chr4.178946041_A = sample(chr4.178946041_A), 
         chr7.8599340_A = sample(chr7.8599340_A)) %>% 
  mutate(chr10.23267180_G = sample(chr10.23267180_G), 
         chr3.136492861_G = sample(chr3.136492861_G)) %>% 
  mutate(chr18.32316331_A = sample(chr18.32316331_A), 
         chr19.24321261_T = sample(chr19.24321261_T)) %>% 
  mutate(chr18.27348077_G = sample(chr18.27348077_G), 
         chr1.281788173_G = sample(chr1.281788173_G)) %>% 
  mutate(chr2.241577141_C = sample(chr2.241577141_C),
         chr5.72916242_T = sample(chr5.72916242_T)) %>%
  mutate(chr8.103608382_G = sample(chr8.103608382_G),
         chr9.15866960_A = sample(chr9.15866960_A)) %>% 
  mutate(chr1.203085725_C = sample(chr1.203085725_C), 
         chr10.84091208_T = sample(chr10.84091208_T))

#select random portion of dataset to be ranked and re-ordered with phenotype
clip_vec <- sample(nrow(BMIwTail), nrow(BMIwTail) * 0.40) # number here is how much of the dataset gets ranked
clipped <- BMIwTail[clip_vec, ] #creates dataset to be ranked and re-ordered
clip_leftover <- BMIwTail[-clip_vec, ] #creates leftover portion of to be shuffled

#shuffle leftover portion of dataset
clip_leftover_shuffle <- clip_leftover %>% 
  mutate(chr6.29889998_C = sample(chr6.29889998_C), 
         chr7.129118847_C = sample(chr7.129118847_C)) %>% 
  mutate(chr4.178946041_A = sample(chr4.178946041_A), 
         chr7.8599340_A = sample(chr7.8599340_A)) %>%
  mutate(chr5.107167969_G = sample(chr5.107167969_G), 
         chr5.72916242_T = sample(chr5.72916242_T)) %>% 
  mutate(chr18.32316331_A = sample(chr18.32316331_A), 
         chr19.24321261_T = sample(chr19.24321261_T)) %>% 
  mutate(chr18.27348077_G = sample(chr18.27348077_G), 
         chr10.84091208_T = sample(chr10.84091208_T)) %>% 
  mutate(chr8.103608382_G = sample(chr8.103608382_G), 
         chr9.15866960_A = sample(chr9.15866960_A)) %>% 
  mutate(chr1.203085725_C = sample(chr1.203085725_C), 
         chr1.281788173_G = sample(chr1.281788173_G)) %>% 
  mutate(chr2.241577141_C = sample(chr2.241577141_C), 
         chr3.136492861_G = sample(chr3.136492861_G)) %>% 
  mutate(chr9.71715296_A = sample(chr9.71715296_A), 
         chr10.23267180_G = sample(chr10.23267180_G)) %>% 
  #---REMOVE COLUMNS THAT AREN'T BEING SHUFFLED THEN RANKED---#
    #in this case, only two variables are NOT participating in an interaction
    #the other variables in this dataset will be combined row-wise with the portion of the dataframe including interactions
  select(-chr5.107167969_G, -chr9.71715296_A) 
         #-chr6.29889998_C, -chr7.129118847_C) 
         #-chr4.178946041_A, -chr7.8599340_A) 
         #-chr3.136492861_G, -chr10.23267180_G)
         #-chr18.32316331_A, -chr19.24321261_T)
         #-chr18.27348077_G, -chr1.281788173_G)
         #-chr2.241577141_C, -chr5.72916242_T)
         #-chr8.103608382_G, -chr9.15866960_A) 


############penetrance function############

#returns a dataframe with an additional 'XOR_inter' column with the value of the XOR interaction 
FunXOR <- function(df, X1, X2) {
  df <- df %>%
    mutate(XOR_inter = case_when(
      {{X1}} == 0 & {{X2}} == 0 ~ 0,
      {{X1}} == 0 & {{X2}} == 1 ~ 1,
      {{X1}} == 0 & {{X2}} == 2 ~ 0,
      {{X1}} == 1 & {{X2}} == 0 ~ 1,
      {{X1}} == 1 & {{X2}} == 1 ~ 0,
      {{X1}} == 1 & {{X2}} == 2 ~ 1,
      {{X1}} == 2 & {{X2}} == 0 ~ 0,
      {{X1}} == 2 & {{X2}} == 1 ~ 1,
      {{X1}} == 2 & {{X2}} == 2 ~ 0
    ))
  return(df)
}

############creating interactions############

#each 'shuffled' dataset includes 
  #an XOR interaction term produced from two selected features
  #an inter_rank term that ranks the XOR interaction term from 0 to 1
    #ranking ties are broken randomly
    #all XOR interactions of 0 are randomly ranked with lower values
    #then all XOR interactions of 1 are randomly ranked with highger values
  #each 'x_ranked' dataset is ordered by the XOR rank
    #the first two columns are the original features
    #the third column is the XOR interaction term
    #the fourth column is the rank of the XOR interaction term
  #each interaction is labelled with the order in which it was removed in experiment 3
    #we have left the interactions that are NOT used in the code for reference
    #the dataframes produced for those interactions are NOT used when producing the final dataframe (see code below for the area to remove them)

##1
#SECOND TO BE REMOVED
shuffled1 <- clipped %>%
  mutate(chr6.29889998_C = sample(chr6.29889998_C),
         chr7.129118847_C = sample(chr7.129118847_C)) %>%
  FunXOR(chr6.29889998_C, chr7.129118847_C) %>%
  mutate(inter_rank = rank(XOR_inter, ties.method = "random"))

x_ranked1 <- shuffled1[order(shuffled1$inter_rank),
                           c('chr6.29889998_C', 'chr7.129118847_C',
                             'XOR_inter', 'inter_rank')]

##2 
#THIRD TO BE REMOVED
shuffled2 <- clipped %>%
  mutate(chr4.178946041_A = sample(chr4.178946041_A),
         chr7.8599340_A = sample(chr7.8599340_A)) %>%
  FunXOR(chr4.178946041_A, chr7.8599340_A) %>%
  mutate(inter_rank = rank(XOR_inter, ties.method = "random"))

x_ranked2 <- shuffled2[order(shuffled2$inter_rank),
                       c('chr4.178946041_A', 'chr7.8599340_A',
                         'XOR_inter', 'inter_rank')]

##3
#FIFTH TO BE REMOVED
shuffled3 <- clipped %>%
  mutate(chr18.32316331_A = sample(chr18.32316331_A),
         chr19.24321261_T = sample(chr19.24321261_T)) %>%
  FunXOR(chr18.32316331_A, chr19.24321261_T) %>%
  mutate(inter_rank = rank(XOR_inter, ties.method = "random"))

x_ranked3 <- shuffled3[order(shuffled3$inter_rank),
                       c('chr18.32316331_A', 'chr19.24321261_T',
                         'XOR_inter', 'inter_rank')]

##4
#SEVENTH TO BE REMOVED
shuffled4 <- clipped %>%
  mutate(chr2.241577141_C = sample(chr2.241577141_C),
         chr5.72916242_T = sample(chr5.72916242_T)) %>%
  FunXOR(chr2.241577141_C, chr5.72916242_T) %>%
  mutate(inter_rank = rank(XOR_inter, ties.method = "random"))

x_ranked4 <- shuffled4[order(shuffled4$inter_rank),
                       c('chr2.241577141_C', 'chr5.72916242_T',
                         'XOR_inter', 'inter_rank')]

##5
#SIXTH TO BE REMOVED
shuffled5 <- clipped %>%
  mutate(chr18.27348077_G = sample(chr18.27348077_G),
         chr1.281788173_G = sample(chr1.281788173_G)) %>%
  FunXOR(chr18.27348077_G, chr1.281788173_G) %>%
  mutate(inter_rank = rank(XOR_inter, ties.method = "random"))

x_ranked5 <- shuffled5[order(shuffled5$inter_rank),
                       c('chr18.27348077_G', 'chr1.281788173_G',
                         'XOR_inter', 'inter_rank')]

##6
#EIGHT TO BE REMOVED
shuffled6 <- clipped %>%
  mutate(chr8.103608382_G = sample(chr8.103608382_G),
         chr9.15866960_A = sample(chr9.15866960_A)) %>%
  FunXOR(chr8.103608382_G, chr9.15866960_A) %>%
  mutate(inter_rank = rank(XOR_inter, ties.method = "random"))

x_ranked6 <- shuffled6[order(shuffled6$inter_rank),
                       c('chr8.103608382_G', 'chr9.15866960_A',
                         'XOR_inter', 'inter_rank')]

##7
#NINTH TO BE REMOVED
shuffled7 <- clipped %>% 
  mutate(chr1.203085725_C = sample(chr1.203085725_C), 
         chr10.84091208_T = sample(chr10.84091208_T)) %>% 
  FunXOR(chr1.203085725_C, chr10.84091208_T) %>% 
  mutate(inter_rank = rank(XOR_inter, ties.method = "random")) 

x_ranked7 <- shuffled7[order(shuffled7$inter_rank), 
                       c('chr1.203085725_C', 'chr10.84091208_T', 
                         'XOR_inter', 'inter_rank')]

##8
#FIRST TO BE REMOVED
shuffled8 <- clipped %>%
  mutate(chr5.107167969_G = sample(chr5.107167969_G),
         chr9.71715296_A = sample(chr9.71715296_A)) %>%
  FunXOR(chr5.107167969_G, chr9.71715296_A) %>%
  mutate(inter_rank = rank(XOR_inter, ties.method = "random"))

x_ranked8 <- shuffled8[order(shuffled8$inter_rank),
                       c('chr5.107167969_G', 'chr9.71715296_A',
                         'XOR_inter', 'inter_rank')]

##9
#FOURTH TO BE REMOVED
shuffled9 <- clipped %>%
  mutate(chr10.23267180_G = sample(chr10.23267180_G),
         chr3.136492861_G = sample(chr3.136492861_G)) %>%
  FunXOR(chr10.23267180_G, chr3.136492861_G) %>%
  mutate(inter_rank = rank(XOR_inter, ties.method = "random"))

x_ranked9 <- shuffled9[order(shuffled9$inter_rank),
                       c('chr10.23267180_G', 'chr3.136492861_G',
                         'XOR_inter', 'inter_rank')]

############re-combining shuffled and ranked data############

#combine ranked phenotype with ranked features, column-wise
#---REMOVE X_RANKED DATAFRAMES WITH INTERACTIONS THAT AREN'T USED---#
  #commenting these dataframes out will prevent those interactions from being included in the final dataset
clipped_final <- clipped[order(clipped$y_rank),] %>% #order phenotypes by their ranking
  select(bmi_w_tail_res_res, y_rank) %>% #only select phenotype and y_rank
  #the first and second columns of each x_ranked data include 2 features, ranked by their XOR interaction
  cbind(x_ranked1[, c(1, 2)]) %>%   #SECOND TO BE REMOVED
  cbind(x_ranked2[, c(1, 2)]) %>%   #THIRD TO BE REMOVED
  cbind(x_ranked3[, c(1, 2)]) %>%   #FIFTH TO BE REMOVED
  cbind(x_ranked4[, c(1, 2)]) %>%   #SEVENTH TO BE REMOVED
  cbind(x_ranked5[, c(1, 2)]) %>%   #SIXTH TO BE REMOVED
  cbind(x_ranked6[, c(1, 2)]) %>%   #EIGHTH TO BE REMOVED
  cbind(x_ranked7[, c(1, 2)]) %>%   #NINTH TO BE REMOVED
  #cbind(x_ranked8[, c(1, 2)]) %>%  #FIRST TO BE REMOVED
  cbind(x_ranked9[, c(1, 2)]) %>%   #FOURTH TO BE REMOVED
  select(colnames(clip_leftover_shuffle)) #re-order columns to original order

#combine row-wise with leftover shuffled data
comb_final <- clipped_final %>% 
  rbind(clip_leftover_shuffle)

#order by y_rank
comb_final <- comb_final[order(comb_final$y_rank), ]

#combine column-wise with features that did NOT participate in interactions
#---SELECT COLUMNS THAT DID NOT PARTICIPATE IN INTERACTIONS TO BE RE-COMBINED---#
comb_shuff_final <- y_rankedBMIwTail_shuff %>% 
  select(chr5.107167969_G, chr9.71715296_A) %>% 
         #chr6.29889998_C, chr7.129118847_C) %>% 
         #chr4.178946041_A, chr7.8599340_A) %>% 
         #chr10.23267180_G, chr3.136492861_G) %>% 
         #chr18.32316331_A, chr19.24321261_T) %>% 
         #chr18.27348077_G, chr1.281788173_G) %>% 
         #chr2.241577141_C, chr5.72916242_T) %>%  
         #chr8.103608382_G, chr9.15866960_A) %>% 
  cbind(comb_final) %>% 
  select(colnames(BMIwTail)) #re-order columns to original order

write_csv(comb_shuff_final[,1:19], "8interactions_data.csv") #don't include y_rank in final csv

#linear model with only univariate effects on shuffled and ranked data
XORmod <- lm(bmi_w_tail_res_res ~ ., data = comb_shuff_final[, 1:19])
summary(XORmod)



