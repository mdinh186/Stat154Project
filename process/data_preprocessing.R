NA_factor <- function(data){
  #Dealing with NAs
  data[data == " ?"] <- NA #Change all '?' values to NA
  NA_Columns <- colSums(is.na(data)) #per column
  NA_percent <- colMeans(is.na(data)*100) #percents
  
  #Convert back to factors
  data$"workclass" <- as.factor(data$"workclass")
  data$"occupation" <- as.factor(data$"occupation")
  data$"native_country" <- as.factor(data$"native_country")
  data
}

NA_imputes <- function(data){
  #Imputation of factorized data
  imputationResults <- missForest(xmis = data, maxiter = 2, variablewise = TRUE)
  dataMissForestImputed <- imputationResults$ximp #this is the adult data set with all the imputations
}


combine_education = function(data){
  dat <- data.table(data)
  dat$education = as.character(dat$education)
  level1 = c(" Preschool"," 1st-4th", " 5th-6th", " 7th-8th")
  dat[,education := ifelse(education %in% level1, "Less than highschool",education)] 
  level2 = c(" 9th", " 10th", " 11th", " 12th")
  dat[,education := ifelse(education %in% level2, "HS with no degree",education)] 
  dat[,education := ifelse(education == " HS-grad", "HS",education)] 
  level3 = c(" Assoc-acdm"," Assoc-voc")
  dat[,education := ifelse(education %in% level3, "Associate",education)] 
  level4 = c(" Prof-school", " Doctorate")
  dat[,education := ifelse(education %in% level4, "Doctorate or Professional",education)] 
  return (dat)
}

education_median = function(data){
  dat <- data.table(data)
  degree = sort(unique(dat$education))
  degree_median_inc = data.frame(education = degree, 
                                 Edu_Mean_inc = c(52857, 64960,37156,42118,81007,31376,18298,15043))
  dat = merge(dat, degree_median_inc, by = "education")
  dat$education = NULL
  return (dat)
}

age_combine <- function(data){
  data$'age' <- cut(data$age, c(seq(14,74, 10), 90), labels = 1:7)
  data$'age' <- as.factor(data$'age')
  data
}

age_median = function(dat){
  Age_med_inc = c(19340, 33151, 41667, 47261, 35232,  21422, 14731)
  xf <- as.factor(dat$age)
  dat$age <- Age_med_inc[xf]
  dat
}

race_combine = function(data){
  dat <- data.table(data)
  dat$race = as.character(dat$race)
  race = unique(dat$race)
  dat[, race := ifelse(race ==" Amer-Indian-Eskimo", " Other", race)] 
  dat$sex = as.character(dat$sex)
  sex = unique(dat$sex)
  gen_race = rep("NA", nrow(dat))
  for (i in 1:nrow(dat)){
    for (r in race){
      if (dat$race[i] == r){
        if (dat$sex[i] == " Female"){
          gen_race[i] = paste0("f", substr(r, 1, 3))
        } else{
          gen_race[i] = paste0("m", substr(r, 1, 3))
        }
      }
    }
  }
  gen_race = gsub(gen_race, pattern = " ", replacement = "")
  dat$gen_race = gen_race
  return (dat)
}

marital_combine <- function(data){
  data <- as.data.frame(data)
  data$marital_status = gsub(
    "^ Divorced","No-Spouse",data$marital_status)
  data$marital_status = gsub(
    "^ Widowed","No-Spouse",data$marital_status)
  data$marital_status = gsub(
    "^ Married-spouse-absent","Absent-Spouse",data$marital_status)
  data$marital_status = gsub(
    "^ Never-married","Single",data$marital_status)
  data$marital_status = gsub(
    "^ Separated","Absent-Spouse",data$marital_status)
  
  data$marital_status = gsub(
    "^ Married-AF-spouse","Has-Spouse",data$marital_status)
  data$marital_status = gsub(
    "^ Married-civ-spouse","Has-Spouse",data$marital_status)
  data
}

occup_sex_med = function(dat){
  interacts <- interaction(dat$sex,dat$occupation)
  medians <- c(21023, 41410, 10905, 26449, 25980, 7059, 41090, 15880, 25790, 24681, 21546, 10431, 20621, 22524, 30642, 11823, 11746,9070, 15815, 12095, 11686, 7453, 7893, 23656, 3287, 4223, 11746)
  x.lvl <- unique(interacts)
  dat$occ_sex <- medians[match(interacts, x.lvl)]
  dat$sex <- NULL
  dat
}

occupation_combine <- function(data){
  data <- as.data.frame(data)
  data$occupation =   
    gsub("^ Adm-clerical","White Collar Not Skilled",data$occupation)
  data$occupation =   
    gsub("^ Armed-Forces","Military",data$occupation)
  data$occupation =   
    gsub("^ Craft-repair","Blue-Collar Services",data$occupation)
  data$occupation = 
    gsub("^ Exec-managerial","White-Collar Skilled",data$occupation)
  data$occupation = 
    gsub("^ Farming-fishing","Blue-Collar Skilled",data$occupation)
  data$occupation = 
    gsub("^ Handlers-cleaners","Other Services",data$occupation)
  data$occupation = 
    gsub("^ Machine-op-inspct","Blue-Collar Skilled",data$occupation)
  data$occupation = 
    gsub("^ Other-service","Other Services",data$occupation)
  data$occupation = 
    gsub("^ Priv-house-serv","Other Services",data$occupation)
  data$occupation = 
    gsub("^ Prof-specialty","White-Collar Skilled",data$occupation)
  data$occupation = 
    gsub("^ Protective-serv","White Collar Services",data$occupation)
  data$occupation = 
    gsub("^ Sales","White Collar Services",data$occupation)
  data$occupation =   
    gsub("^ Tech-support","White Collar Services",data$occupation)
  data$occupation = 
    gsub("^ Transport-moving","Blue-Collar Services",data$occupation)
  data
} 
country_combine <- function(data){
  data <- as.data.frame(data)
  data$native_country = gsub(
    "^ Cambodia","South Asia, 1",data$native_country)
  data$native_country = gsub(
    "^ Canada","Canada",data$native_country)
  data$native_country = gsub(
    "^ China","China",data$native_country)
  data$native_country = gsub(
    "^ Columbia","Carribean-1",
    data$native_country)
  data$native_country = gsub(
    "^ Cuba","US",data$native_country)
  data$native_country = gsub(
    "^ Dominican-Republic","Carribean-1",
    data$native_country)
  data$native_country = gsub(
    "^ Ecuador","South America",data$native_country)
  data$native_country = gsub(
    "^ El-Salvador","Central-America",
    data$native_country)
  data$native_country = gsub(
    "^ England","England",data$native_country)
  data$native_country = gsub(
    "^ France","Central-West",data$native_country)
  data$native_country = gsub(
    "^ Germany","Central-Europe",data$native_country)
  data$native_country = gsub(
    "^ Greece","Central-Europe",data$native_country)
  data$native_country = gsub(
    "^ Guatemala","Central-America",
    data$native_country)
  data$native_country = gsub(
    "^ Haiti","Carribean-2",data$native_country)
  data$native_country = gsub(
    "^ Holand-Netherlands","Central-West",
    data$native_country)
  data$native_country = gsub(
    "^ Honduras","Central-America",
    data$native_country)
  data$native_country = gsub(
    "^ Hong","China",data$native_country)
  data$native_country = gsub(
    "^ Hungary","East-Europe",data$native_country)
  data$native_country = gsub(
    "^ India","South Asia, 1",data$native_country)
  data$native_country = gsub(
    "^ Iran","South Asia, 1",data$native_country)
  data$native_country = gsub(
    "^ Ireland","Ireland-Scotland",
    data$native_country)
  data$native_country = gsub(
    "^ Italy","Central-Europe",data$native_country)
  data$native_country = gsub(
    "^ Jamaica","Carribean-2",data$native_country)
  data$native_country = gsub(
    "^ Japan","Japan",data$native_country)
  data$native_country = gsub(
    "^ Laos","South Asia, 2",data$native_country)
  data$native_country = gsub(
    "^ Mexico","Central-America",
    data$native_country)
  data$native_country = gsub(
    "^ Nicaragua","Central-America",
    data$native_country)
  data$native_country = gsub(
    "^ Peru","South America",data$native_country)
  data$native_country = gsub(
    "^ Philippines","South Asia, 3",
    data$native_country)
  data$native_country = gsub(
    "^ Poland","East-Europe",data$native_country)
  data$native_country = gsub(
    "^ Portugal","West-Europe",data$native_country)
  data$native_country = gsub(
    "^ Puerto-Rico","Carribean-2",
    data$native_country)
  data$native_country = gsub(
    "^ Scotland","Ireland-Scotland",
    data$native_country)
  data$native_country = gsub(
    "^ South","East-Europe",data$native_country)
  data$native_country = gsub(
    "^ Taiwan","South Asia, 3",data$native_country)
  data$native_country = gsub(
    "^ Thailand","South Asia, 2",data$native_country)
  data$native_country = gsub(
    "^ Trinadad&Tobago","Carribean-2",
    data$native_country)
  data$native_country = gsub(
    "^ United-States","US",data$native_country)
  data$native_country = gsub(
    "^ Vietnam","South Asia, 2",data$native_country)
  data$native_country = gsub(
    "^ Yugoslavia","Central-Europe",data$native_country)
  #Having some issues with this one:
  vecc <- which(data$native_country == 
                  " Outlying-US(Guam-USVI-etc)")
  data$native_country[vecc] <- "Carribean-1"
  data
}
workclass_combine <- function(data){
  data <- as.data.frame(data)
  data$workclass = gsub(
    "^ Federal-gov","Gov",data$workclass)
  data$workclass = gsub(
    "^ Local-gov","Gov",data$workclass)
  data$workclass = gsub(
    "^ State-gov","Gov",data$workclass)
  data$workclass = gsub(
    "^ Private","Private",data$workclass)
  data$workclass = gsub(
    "^ Self-emp-
    ","Self-Employed-
    orp",data$workclass)
  data$workclass = gsub(
    "^ Self-emp-not-
    ","Self-Employed-Not-
    orp",data$workclass)
  data$workclass = gsub(
    "^ Without-pay","Not-Working",data$workclass)
  data$workclass = gsub(
    "^ Never-worked","Not-Working",data$workclass)
  data
}
relation_combine <- function(data){
  data <- as.data.frame(data)
  data$relationship = gsub(
    "^ Husband","Spouse",data$relationship)
  data$relationship = gsub(
    "^ Wife","Spouse",data$relationship)
  data$relationship = gsub(
    "^ Other-relative","Other-relative",data$relationship)
  data$relationship = gsub(
    "^ Own-child","Other-relative",data$relationship)
  data$relationship = gsub(
    "^ Unmarried","Not-relative",data$relationship)
  data$relationship = gsub(
    "^ Not-in-family","Not-relative",data$relationship)
  data
}
cap_gain_clean_outlier <- function(data){
  data <- as.data.frame(data)
  mean_cg <- mean(data$'capital-gain')
  data$`capital-gain`[which(data$`capital-gain` == 99999)] <- mean_cg
  data
} 


factorize <- function(data){
  data <- as.data.frame(data)
  cols <- c("age", "workclass", 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country') #no finl-weight because that is continuous
  data[cols] <- lapply(data[cols], as.factor)
  data
}

mrg_combine = function(data){
  dat <- data.table(data)
  dat$`marital-status` = as.character(dat$`marital-status`)
  mrg = unique(dat$`marital-status`)
  
  dat$sex = as.character(dat$sex)
  sex = unique(dat$sex)
  gen_mrg = rep("NA", nrow(dat))
  for (i in 1:nrow(dat)){
    for (m in mrg){
      if (dat$`marital-status`[i] == m){
        if (dat$sex[i] == " Female"){
          gen_mrg [i] = paste0("f", substr(m, 1, 3))
        } else{
          gen_mrg [i] = paste0("m", substr(m, 1, 3))
        }
      }
    }
  }
  gen_mrg  = gsub(gen_mrg , pattern = " ", replacement = "")
  dat$gen_mrg  = gen_mrg 
  return (dat)
} 


gen_med = function(data){
  gender = sort(unique(data$sex))
  gen_med_inc = data.table(sex = gender, 
                           Gen_Med_Inc = c( 22834,31728))
  gen_mrg = sort(unique(data$gen_mrg))
  gen_mrg_med_inc = data.table(gen_mrg = gen_mrg,Gen_Med_Mrg_Inc = c(20435,46317,15372, 15372,31336,46317,25290,25290))
  data = merge(data, gen_med_inc, by = "sex")
  data = merge(data, gen_mrg_med_inc, by = "gen_mrg")
  data$gen_mrg = NULL
  data$sex = NULL
  return (data)
}



race_med = function(data){
  race = sort(unique(data$race))
  race_med_inc= data.table(race = race, 
                           Race_Med_Inc = c(17381, 10952,9702,18110))
  data = merge(data, race_med_inc, by = "race")
  data$race = NULL
  return(data)
}






###########################################
# use this function for test data 
data_processing = function(data){
  
  ###imputation: 
  NA_factor = 
  
  
  ### feature engineering: 
  data = age_combine(data)
  data = age_median(data)
  data = combine_education(data)
  data = education_median(data)
  data = race_combine(data)
  data = marital_combine(data)
  data = occup_sex_med(data)
  data = occupation_combine(data)
  data = country_combine(data)
  data = workclass_combine(data)
  data = relation_combine(data)
  data = cap_gain_clean_outlier(data)
  data = mrg_combine(data)
  data = gen_med(data)
  data = race_med(data)
  

}
data_processing