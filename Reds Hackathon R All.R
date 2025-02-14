#Cincinnati Reds Hackathon 2025 R Code
#IAA Team
#Andrew Buelna, Landon Docherty, Brett Laderman, Danny Ryan and Jacob Segmiller

#Loading Packages
library(dplyr)
library(tidyr)
library(tidyverse)
library(readxl)
library(stringr)
library(readr)
library(xgboost)
library(stringr)
library(writexl)
library(caret)
library(xgboost)
library(dplyr)
library(data.table)
library(caret)
library(randomForest)
library(doParallel)

#Reading in Data
savant <- read_csv("C:/Users/seggy/Documents/Reds Hackathon 2025/savant_data_2021_2023.csv")
lahman <- read_csv("C:/Users/seggy/Documents/Reds Hackathon 2025/lahman_people.csv")

#Creating Hitters Dataset with One Row per Player per Season
savant <- savant %>% mutate(year = str_sub(game_date, 1, 4),
                            events = ifelse(events == 'caught_stealing_3b' | events == 'caught_stealing_2b' | 
                                              events == 'pickoff_caught_stealing_2b' | events == 'pickoff_2b' |
                                              events == 'pickoff_caught_stealing_3b' | events == 'pickoff_caught_stealing_3b' | 
                                              events == 'pickoff_1b' | events == 'wild_pitch' | events == 'game_advisory' |
                                              events == 'caught_stealing_home' | events == 'pickoff_3b' |
                                              events == 'stolen_base_2b' | events == 'stolen_base_3b' |
                                              events == 'passed_ball' | events == 'pickoff_caught_stealing_home' |
                                              events == 'caught_stealing_home' | events == 'pickoff_error_3b', NA, events))
batters <- savant %>%
  filter(!is.na(events)) %>% 
  mutate(weight = case_when(
    year == 2021 ~ 0.1,
    year == 2022 ~ 0.3,
    year == 2023 ~ 0.6
  )) %>%
  group_by(batter, year) %>%
  summarise(games = n_distinct(game_date),
            PA = n(),
            PA_PG = round(PA / games, 3),
            H = sum(events %in% c("single", "double", "triple", "home_run")),
            Singles = sum(events == "single"),
            Doubles = sum(events == "double"),
            Triples = sum(events == "triple"),
            HR = sum(events == "home_run"),
            SO = sum(events == "strikeout"),
            BB = sum(events == "walk"),
            HBP = sum(events == "hit_by_pitch"),
            SF = sum(events == "sac_fly"),
            SB = sum(events == "sac_bunt"),
            CI = sum(events == "catcher_interf"),
            AB = sum(events %in% c("single", "double", "triple", "home_run", 
                                   "field_out", "strikeout", "field_error", "force_out", "grounded_into_double_play",
                                   "fielders_choice", "other_out", "strikeout_double_play",
                                   "fielders_choice_out", "double_play", "sac_fly_double_play",
                                   "triple_play", "sac_bunt_double_play")),
            BA = round(H / AB, 3),
            OBP = round((H + BB + HBP) / (AB + BB + HBP + SF), 3),
            SLG = round((Singles + 2 * Doubles + 3 * Triples + 4 * HR) / AB, 3),
            OPS = round(OBP + SLG, 3),
            ISO = round(SLG - BA, 3),
            BABIP = round((H - HR) / (AB - SO - HR + SF), 3),
            K_perc = round(SO / PA, 3),
            BB_perc = round(BB / PA, 3),
            HR_perc = round(HR / PA, 3),
            XBH = (Doubles + Triples + HR),
            TB = Singles + 2 * Doubles + 3 * Triples + 4 * HR,
            RC = round(((H + BB) * TB) / PA, 3),
            Contact_perc = round((AB - SO) / AB, 3),
            BB_K_ratio = round(BB / SO, 3),
            RC_per_PA = round(RC / PA, 3),
            XBH_perc = round(XBH / H, 3),
            weight = first(weight))

#Calculate Player Age for Batters
batters <- batters %>%
  left_join(lahman %>% select(player_mlb_id, birthDate), 
            by = c("batter" = "player_mlb_id"))
batters$birthDate <- as.Date(batters$birthDate)
batters <- batters %>%
  mutate(
    age = case_when(
      year == "2021" ~ floor(interval(birthDate, as.Date("2021-04-01")) / years(1)),
      year == "2022" ~ floor(interval(birthDate, as.Date("2022-04-01")) / years(1)),
      year == "2023" ~ floor(interval(birthDate, as.Date("2023-04-01")) / years(1)),
      TRUE ~ NA_real_
    )
  )

#Calculate Weighted Spot Average
#Getting Rid of Rows w/o player_mlb_id
lahman1 <- lahman %>% filter(nchar(player_mlb_id) > 2)
#Adding Danny's New ID to Lahman
lahman2 <- lahman1 %>% mutate(batterID = paste("player", row_number(), sep = "_"))
#Creating a Dataset of Only player_mlb_id and batterID
lahman3 <- lahman2 %>% select(player_mlb_id, batterID)
#Same as Lahman3 but Says PitcherID
lahman4 <- lahman3 %>% rename(pitcherID = batterID)
#Making a dataset of All 30 Teams
team1 <- savant %>%  distinct(home_team) %>% mutate(teamID = paste("team", row_number(), sep = "_"))
#Splitting The Teams by Home and Away
team1_home <- team1 %>% rename(teamID_home = teamID)
team1_away <- team1 %>% rename(teamID_away = teamID, away_team = home_team)
#Merging the New ID onto the Savant Dataset
savant1 <- left_join(savant, lahman3, by = c("batter" = "player_mlb_id"))
#Updating Home Team
savant2 <- left_join(savant1,team1_home, by = "home_team" )
#Updating Away Team
savant3 <- left_join(savant2, team1_away, by = "away_team")
#Adding Pitcher ID
savant4 <- left_join(savant3, lahman4, by = c("pitcher" = "player_mlb_id"))
#Testing Batting Order for One Game
savant5 <- savant4 %>% mutate(events = ifelse(events == 'caught_stealing_3b' | events == 'caught_stealing_2b' |
                                                events == 'pickoff_caught_stealing_2b' | events == 'pickoff_2b' |
                                                events == 'pickoff_caught_stealing_3b' | events == 'pickoff_caught_stealing_3b' |
                                                events == 'pickoff_1b' | events == 'wild_pitch' | events == 'game_advisory' |
                                                events == 'caught_stealing_home' | events == 'pickoff_3b' |
                                                events == 'stolen_base_2b' | events == 'stolen_base_3b' |
                                                events == 'passed_ball' | events == 'pickoff_caught_stealing_home' |
                                                events == 'caught_stealing_home' | events == 'pickoff_error_3b', NA, events))
#Subsetting to One Row Per AB
savant6 <- savant5 %>% filter(nchar(events) > 2)
#Putting Two New Columns in Per Row
savant7 <- savant6 %>% arrange(game_pk,inning_topbot, at_bat_number) %>% 
  group_by(game_pk, inning_topbot) %>% 
  mutate(rank = min_rank(at_bat_number)) %>% 
  ungroup() %>% 
  mutate(spot = ifelse(rank %% 9 == 0, 9,rank %% 9))
#Adding Year
savant7 <- savant7 %>% mutate(year = str_sub(game_date,1,4))
#Getting Number of AB's per Spot and Percentage of AB's per Spot by Player and Year
savant8 <- savant7 %>% mutate(
  temp01 = ifelse(spot == 1, 1,0),
  temp02 = ifelse(spot == 2, 1,0),
  temp03 = ifelse(spot == 3, 1,0),
  temp04 = ifelse(spot == 4, 1,0),
  temp05 = ifelse(spot == 5, 1,0),
  temp06 = ifelse(spot == 6, 1,0),
  temp07 = ifelse(spot == 7, 1,0),
  temp08 = ifelse(spot == 8, 1,0),
  temp09 = ifelse(spot == 9, 1,0))
#Summarising by Player by Year
savant9 <- savant8 %>% group_by(batterID,batter, year) %>% 
  summarise(
    #summing batting spot per year and per batter
    spot01 = sum(temp01),
    spot02 = sum(temp02),
    spot03 = sum(temp03),
    spot04 = sum(temp04),
    spot05 = sum(temp05),
    spot06 = sum(temp06),
    spot07 = sum(temp07),
    spot08 = sum(temp08),
    spot09 = sum(temp09),
    #total at bats per year per player
    AB = sum(spot01 + spot02 + spot03 + spot04 + spot05 + spot06 + spot07 + spot08 +spot09),
    #percent of AB's per year in each batting spot
    spot01_per = (spot01/AB)*100,
    spot02_per = (spot02/AB)*100,
    spot03_per = (spot03/AB)*100,
    spot04_per = (spot04/AB)*100,
    spot05_per = (spot05/AB)*100,
    spot06_per = (spot06/AB)*100,
    spot07_per = (spot07/AB)*100,
    spot08_per = (spot08/AB)*100,
    spot09_per = (spot09/AB)*100) %>% 
  #weighted value representing spot hit in order
  mutate(
    weighted_spot_avg = (
      ((1*spot01_per) +
         (2*spot02_per) + 
         (3*spot03_per) +
         (4*spot04_per) +
         (5*spot05_per) +
         (6*spot06_per) +
         (7*spot07_per) +
         (8*spot08_per) +
         (9*spot09_per))/100))

#Join weighted_spot_avg to Batters
batters <- left_join(batters, savant9 %>% select(batter, year, weighted_spot_avg), by = c("batter", "year"))
batters <- batters %>% 
  select(-batterID)
#write.csv(batters, "C:/Users/seggy/Documents/Reds Hackathon 2025/batters.csv")

#Batters Training/Testing Split
train <- batters %>% 
  filter(year == "2021" | year == "2022")
test <- batters %>% 
  filter(year == "2023")
#write.csv(train, "C:/Users/seggy/Documents/Reds Hackathon 2025/batters_train.csv")
#write.csv(test, "C:/Users/seggy/Documents/Reds Hackathon 2025/batters_test.csv")

#Batters 2024 Stats
batters_future <- batters %>% 
  group_by(batter) %>%
  summarise(games = sum(games * weight),
            PA = sum(PA * weight),
            PA_PG = round(sum(PA_PG * weight), 3),
            H = sum(H * weight),
            Singles = sum(Singles * weight),
            Doubles = sum(Doubles * weight),
            Triples = sum(Triples * weight),
            HR = sum(HR * weight),
            SO = sum(SO * weight),
            BB = sum(BB * weight),
            HBP = sum(HBP * weight),
            SF = sum(SF * weight),
            SB = sum(SB * weight),
            CI = sum(CI * weight),
            AB = sum(AB * weight),
            BA = round(sum(H * weight) / sum(AB * weight), 3),
            OBP = round((sum(H * weight) + sum(BB * weight) + sum(HBP * weight)) / 
                          (sum(AB * weight) + sum(BB * weight) + sum(HBP * weight) + sum(SF * weight)), 3),
            SLG = round(sum(TB * weight) / sum(AB * weight), 3),
            OPS = round(sum(OBP * weight) + sum(SLG * weight), 3),
            ISO = round(sum(SLG * weight) - sum(BA * weight), 3),
            BABIP = round((sum(H * weight) - sum(HR * weight)) / 
                            (sum(AB * weight) - sum(SO * weight) - sum(HR * weight) + sum(SF * weight)), 3),
            K_perc = round(sum(SO * weight) / sum(PA * weight), 3),
            BB_perc = round(sum(BB * weight) / sum(PA * weight), 3),
            HR_perc = round(sum(HR * weight) / sum(PA * weight), 3),
            XBH = sum(XBH * weight),
            TB = sum(TB * weight),
            RC = round(sum(((H + BB) * TB) * weight) / sum(PA * weight), 3),
            Contact_perc = round(sum((AB - SO) * weight) / sum(AB * weight), 3),
            BB_K_ratio = round(sum(BB * weight) / sum(SO * weight), 3),
            RC_per_PA = round(sum(RC * weight) / sum(PA * weight), 3),
            XBH_perc = round(sum(XBH * weight) / sum(H * weight), 3),)

#Calculate Player Age for Batters 2024
#batters_future <- read_csv("C:/Users/seggy/Documents/Reds Hackathon 2025/battingstats_future.csv")
batters_future <- batters_future %>% 
  left_join(lahman %>% select(player_mlb_id, birthDate), 
            by = c("batter" = "player_mlb_id")) %>% 
  mutate(year = "2024") %>% 
  mutate(age = case_when(
         year == "2024" ~ floor(interval(birthDate, as.Date("2024-04-01")) / years(1)),
         TRUE ~ NA_real_)
)
#write.csv(batters_future, "C:/Users/seggy/Documents/Reds Hackathon 2025/battingstate_future_age.csv")

#Pitcher Cleaning
savant <- savant %>% mutate(year = str_sub(game_date,1,4))
#2021
savant_2021 <- savant %>% filter(year == "2021")
#2022
savant_2022 <- savant %>% filter(year == "2022")
#2023
savant_2023 <- savant %>% filter(year == "2023")
# Roll up pitcher data on game-by-game stats
# 2021
pitcher_game_21 <- savant_2021 %>% group_by(pitcher,game_pk,pitch_type) %>% 
  summarize(release_speed=mean(release_speed,na.rm=T),
            release_x=mean(release_pos_x,na.rm=T),
            release_y=mean(release_pos_y,na.rm=T),
            release_z=mean(release_pos_z,na.rm=T),
            zone=mean(zone,na.rm=T), strikes=sum(type=='S'), 
            balls=sum(type=='B'), line_drives=sum(bb_type=='line_drive'),
            fly_balls=sum(bb_type=='fly_ball'),
            ground_balls=sum(bb_type=='ground_ball'), 
            popups=sum(bb_type=='popup'),
            movement_horiz=mean(pfx_x,na.rm=T), movement_vert=mean(pfx_z,na.rm=T),
            plate_pos_horiz=mean(plate_x,na.rm=T), plate_pos_vert=mean(plate_z,na.rm=T),
            velocity_x=mean(vx0,na.rm=T), velocity_y=mean(vy0,na.rm=T),
            velocity_z=mean(vz0,na.rm=T), accel_x=mean(ax,na.rm=T),
            accel_y=mean(ay,na.rm=T), accel_z=mean(az,na.rm=T),
            launch_angle=mean(launch_angle,na.rm=T),
            launch_speed=mean(launch_angle,na.rm=T),
            effective_speed=mean(effective_speed,na.rm=T),
            spin_rate=mean(release_spin_rate,na.rm=T),
            extension=mean(release_extension,na.rm=T),
            estimated_BA=mean(estimated_ba_using_speedangle,na.rm=T),
            estimated_wOBA=mean(estimated_woba_using_speedangle,na.rm=T),
            wOBA_value=mean(woba_value,na.rm=T), BABIP_value=mean(babip_value,na.rm=T),
            ISO_value=mean(iso_value,na.rm=T), weak_contact=sum(launch_speed_angle==1),
            topped_contact=sum(launch_speed_angle==2),
            under_contact=sum(launch_speed_angle==3),
            burner_contact=sum(launch_speed_angle==4),
            solid_contact=sum(launch_speed_angle==5),
            barreled_contact=sum(launch_speed_angle==6),
            sp=mean(sp_indicator,na.rm=T),rp=mean(rp_indicator,na.rm=T)) %>%
  mutate(pitches_thrown=strikes+balls+line_drives+fly_balls+ground_balls+popups)
# 2022
pitcher_game_22 <- savant_2022 %>% group_by(pitcher,game_pk,pitch_type) %>% 
  summarize(release_speed=mean(release_speed,na.rm=T),
            release_x=mean(release_pos_x,na.rm=T),
            release_y=mean(release_pos_y,na.rm=T),
            release_z=mean(release_pos_z,na.rm=T),
            zone=mean(zone,na.rm=T), strikes=sum(type=='S'), 
            balls=sum(type=='B'), line_drives=sum(bb_type=='line_drive'),
            fly_balls=sum(bb_type=='fly_ball'),
            ground_balls=sum(bb_type=='ground_ball'), 
            popups=sum(bb_type=='popup'),
            movement_horiz=mean(pfx_x,na.rm=T), movement_vert=mean(pfx_z,na.rm=T),
            plate_pos_horiz=mean(plate_x,na.rm=T), plate_pos_vert=mean(plate_z,na.rm=T),
            velocity_x=mean(vx0,na.rm=T), velocity_y=mean(vy0,na.rm=T),
            velocity_z=mean(vz0,na.rm=T), accel_x=mean(ax,na.rm=T),
            accel_y=mean(ay,na.rm=T), accel_z=mean(az,na.rm=T),
            launch_angle=mean(launch_angle,na.rm=T),
            launch_speed=mean(launch_angle,na.rm=T),
            effective_speed=mean(effective_speed,na.rm=T),
            spin_rate=mean(release_spin_rate,na.rm=T),
            extension=mean(release_extension,na.rm=T),
            estimated_BA=mean(estimated_ba_using_speedangle,na.rm=T),
            estimated_wOBA=mean(estimated_woba_using_speedangle,na.rm=T),
            wOBA_value=mean(woba_value,na.rm=T), BABIP_value=mean(babip_value,na.rm=T),
            ISO_value=mean(iso_value,na.rm=T), weak_contact=sum(launch_speed_angle==1),
            topped_contact=sum(launch_speed_angle==2),
            under_contact=sum(launch_speed_angle==3),
            burner_contact=sum(launch_speed_angle==4),
            solid_contact=sum(launch_speed_angle==5),
            barreled_contact=sum(launch_speed_angle==6),
            sp=mean(sp_indicator,na.rm=T),rp=mean(rp_indicator,na.rm=T)) %>%
  mutate(pitches_thrown=strikes+balls+line_drives+fly_balls+ground_balls+popups)
# 2023
pitcher_game_23 <- savant_2023 %>% group_by(pitcher,game_pk,pitch_type) %>% 
  summarize(release_speed=mean(release_speed,na.rm=T),
            release_x=mean(release_pos_x,na.rm=T),
            release_y=mean(release_pos_y,na.rm=T),
            release_z=mean(release_pos_z,na.rm=T),
            zone=mean(zone,na.rm=T), strikes=sum(type=='S'), 
            balls=sum(type=='B'), line_drives=sum(bb_type=='line_drive'),
            fly_balls=sum(bb_type=='fly_ball'),
            ground_balls=sum(bb_type=='ground_ball'), 
            popups=sum(bb_type=='popup'),
            movement_horiz=mean(pfx_x,na.rm=T), movement_vert=mean(pfx_z,na.rm=T),
            plate_pos_horiz=mean(plate_x,na.rm=T), plate_pos_vert=mean(plate_z,na.rm=T),
            velocity_x=mean(vx0,na.rm=T), velocity_y=mean(vy0,na.rm=T),
            velocity_z=mean(vz0,na.rm=T), accel_x=mean(ax,na.rm=T),
            accel_y=mean(ay,na.rm=T), accel_z=mean(az,na.rm=T),
            launch_angle=mean(launch_angle,na.rm=T),
            launch_speed=mean(launch_angle,na.rm=T),
            effective_speed=mean(effective_speed,na.rm=T),
            spin_rate=mean(release_spin_rate,na.rm=T),
            extension=mean(release_extension,na.rm=T),
            estimated_BA=mean(estimated_ba_using_speedangle,na.rm=T),
            estimated_wOBA=mean(estimated_woba_using_speedangle,na.rm=T),
            wOBA_value=mean(woba_value,na.rm=T), BABIP_value=mean(babip_value,na.rm=T),
            ISO_value=mean(iso_value,na.rm=T), weak_contact=sum(launch_speed_angle==1),
            topped_contact=sum(launch_speed_angle==2),
            under_contact=sum(launch_speed_angle==3),
            burner_contact=sum(launch_speed_angle==4),
            solid_contact=sum(launch_speed_angle==5),
            barreled_contact=sum(launch_speed_angle==6),
            sp=mean(sp_indicator,na.rm=T),rp=mean(rp_indicator,na.rm=T)) %>%
  mutate(pitches_thrown=strikes+balls+line_drives+fly_balls+ground_balls+popups)
# Roll up game-by-game to year-by-year data
# 2021
pitcher_21 <- pitcher_game_21 %>% group_by(pitcher,pitch_type) %>% 
  summarize(release_speed=mean(release_speed,na.rm=T),
            zone=mean(zone,na.rm=T),
            strikes=sum(strikes),
            balls=sum(balls),
            line_drives=sum(line_drives),
            fly_balls=sum(fly_balls),
            ground_balls=sum(ground_balls),
            popups=sum(popups),
            movement_horiz=mean(movement_horiz,na.rm=T),
            movement_vert=mean(movement_vert,na.rm=T),
            velocity_x=mean(velocity_x,na.rm=T),
            velocity_y=mean(velocity_y,na.rm=T),
            velocity_z=mean(velocity_z,na.rm=T),
            accel_x=mean(accel_x,na.rm=T),
            accel_y=mean(accel_y,na.rm=T),
            accel_z=mean(accel_z,na.rm=T),
            launch_angle=mean(launch_angle,na.rm=T),
            launch_speed=mean(launch_speed,na.rm=T),
            effective_speed=mean(effective_speed,na.rm=T),
            spin_rate=mean(spin_rate,na.rm=T),
            wOBA_value=mean(wOBA_value,na.rm=T), BABIP_value=mean(BABIP_value,na.rm=T),
            ISO_value=mean(ISO_value,na.rm=T), weak_contact=sum(weak_contact),
            topped_contact=sum(topped_contact),
            under_contact=sum(under_contact),
            burner_contact=sum(burner_contact),
            solid_contact=sum(solid_contact),
            barreled_contact=sum(barreled_contact),
            sp=mean(sp,na.rm=T),rp=mean(rp,na.rm=T),
            pitches_thrown=sum(pitches_thrown))
# 2022
pitcher_22 <- pitcher_game_22 %>% group_by(pitcher,pitch_type) %>% 
  summarize(release_speed=mean(release_speed,na.rm=T),
            zone=mean(zone,na.rm=T),
            strikes=sum(strikes),
            balls=sum(balls),
            line_drives=sum(line_drives),
            fly_balls=sum(fly_balls),
            ground_balls=sum(ground_balls),
            popups=sum(popups),
            movement_horiz=mean(movement_horiz,na.rm=T),
            movement_vert=mean(movement_vert,na.rm=T),
            velocity_x=mean(velocity_x,na.rm=T),
            velocity_y=mean(velocity_y,na.rm=T),
            velocity_z=mean(velocity_z,na.rm=T),
            accel_x=mean(accel_x,na.rm=T),
            accel_y=mean(accel_y,na.rm=T),
            accel_z=mean(accel_z,na.rm=T),
            launch_angle=mean(launch_angle,na.rm=T),
            launch_speed=mean(launch_speed,na.rm=T),
            effective_speed=mean(effective_speed,na.rm=T),
            spin_rate=mean(spin_rate,na.rm=T),
            wOBA_value=mean(wOBA_value,na.rm=T), BABIP_value=mean(BABIP_value,na.rm=T),
            ISO_value=mean(ISO_value,na.rm=T), weak_contact=sum(weak_contact),
            topped_contact=sum(topped_contact),
            under_contact=sum(under_contact),
            burner_contact=sum(burner_contact),
            solid_contact=sum(solid_contact),
            barreled_contact=sum(barreled_contact),
            sp=mean(sp,na.rm=T),rp=mean(rp,na.rm=T),
            pitches_thrown=sum(pitches_thrown))
# 2023
pitcher_23 <- pitcher_game_23 %>% group_by(pitcher,pitch_type) %>% 
  summarize(release_speed=mean(release_speed,na.rm=T),
            zone=mean(zone,na.rm=T),
            strikes=sum(strikes),
            balls=sum(balls),
            line_drives=sum(line_drives),
            fly_balls=sum(fly_balls),
            ground_balls=sum(ground_balls),
            popups=sum(popups),
            movement_horiz=mean(movement_horiz,na.rm=T),
            movement_vert=mean(movement_vert,na.rm=T),
            velocity_x=mean(velocity_x,na.rm=T),
            velocity_y=mean(velocity_y,na.rm=T),
            velocity_z=mean(velocity_z,na.rm=T),
            accel_x=mean(accel_x,na.rm=T),
            accel_y=mean(accel_y,na.rm=T),
            accel_z=mean(accel_z,na.rm=T),
            launch_angle=mean(launch_angle,na.rm=T),
            launch_speed=mean(launch_speed,na.rm=T),
            effective_speed=mean(effective_speed,na.rm=T),
            spin_rate=mean(spin_rate,na.rm=T),
            wOBA_value=mean(wOBA_value,na.rm=T), BABIP_value=mean(BABIP_value,na.rm=T),
            ISO_value=mean(ISO_value,na.rm=T), weak_contact=sum(weak_contact),
            topped_contact=sum(topped_contact),
            under_contact=sum(under_contact),
            burner_contact=sum(burner_contact),
            solid_contact=sum(solid_contact),
            barreled_contact=sum(barreled_contact),
            sp=mean(sp,na.rm=T),rp=mean(rp,na.rm=T),
            pitches_thrown=sum(pitches_thrown))
# Batters Faced by Year
# 2021
batters_faced_21 <- savant_2021 %>% group_by(pitcher,game_pk,role_key) %>% summarize(batters_faced=max(pitcher_at_bat_number)) %>% 
  group_by(pitcher,role_key) %>% 
  summarize(batters_faced=sum(batters_faced))
# 2022
batters_faced_22 <- savant_2022 %>% group_by(pitcher,game_pk,role_key) %>% summarize(batters_faced=max(pitcher_at_bat_number)) %>% 
  group_by(pitcher,role_key) %>% 
  summarize(batters_faced=sum(batters_faced))
# 2023
batters_faced_23 <- savant_2023 %>% group_by(pitcher,game_pk,role_key) %>% summarize(batters_faced=max(pitcher_at_bat_number)) %>% 
  group_by(pitcher,role_key) %>% 
  summarize(batters_faced=sum(batters_faced))
# Join onto yearly stats
pitcher_21 <- left_join(pitcher_21,batters_faced_21,by="pitcher")
pitcher_22 <- left_join(pitcher_22,batters_faced_22,by="pitcher")
pitcher_23 <- left_join(pitcher_23,batters_faced_23,by="pitcher")
# Season stats 
# Defining out groups 
# 1 out
single_outs <- c("caught_stealing_home","pickoff_3b","pickoff_1b",
                 "pickoff_caught_stealing_3b","pickoff_caught_stealing_2b",
                 "fielders_choice_out","other_out","sac_bunt","sac_fly",
                 "force_out","caught_stealing_3b","strikeout")
# 2 out
double_outs <- c("grounded_into_double_play","strikeout_double_play",
                 "double_play","sac_bunt_double_play")
# 2021 stats
pitcher_stats_21 <- savant_2021 %>%
  # Group by pitcher, game to isolate unique appearances
  group_by(pitcher, game_pk) %>%
  arrange(game_pk, pitch_number_appearance) %>%
  # Calculate the change in outs between consecutive rows
  mutate(out_diff = c(0,diff(outs_when_up))) %>%
  # Correct for inning transitions by handling negatives in outdiff
  mutate(out_diff = ifelse(out_diff == -1, 2, 
                           ifelse(out_diff==-2,1,out_diff))) %>% 
  # Check for outs in last pitch of appearance
  mutate(out_diff=
           ifelse(pitch_number_appearance!=max(pitch_number_appearance),
                  out_diff,
                  ifelse(events %in% single_outs,1,
                         ifelse(events %in% double_outs,2,
                                ifelse(events == "triple_play",3,
                                       out_diff))))) %>% 
  # Now group by pitcher and role key (SP/RP)
  ungroup() %>% 
  group_by(pitcher,role_key) %>%
  # Calculate stats 
  summarize(
    IP = sum(out_diff, na.rm = TRUE) / 3,
    WHIP = sum(events %in% c("walk", "single", "double", "home_run", 
                             "triple")) / 
      (sum(out_diff, na.rm = TRUE) / 3),
    H = sum(events %in% c("single", "double", "home_run", "triple")),
    HR = sum(events == "home_run"),
    ER = sum(post_bat_score - bat_score),
    SO = sum(events %in% c("strikeout", "strikeout_double_play")),
    BB = sum(events == "walk"),
    HBP = sum(events == "hit_by_pitch"),
    # Per 9 innings stats
    ERA = (sum(post_bat_score - bat_score) / 
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    H9 = (sum(events %in% c("single", "double", "home_run", "triple")) / 
            (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    HR9 = (sum(events == "home_run") / 
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    BB9 = (sum(events == "walk") / 
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    SO9 = (sum(events %in% c("strikeout", "strikeout_double_play")) /
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    # SOs/BBs
    SOBBratio = sum(events %in% c("strikeout", "strikeout_double_play")) /
      sum(events == "walk")
  )
# 2022 stats
pitcher_stats_22 <- savant_2022 %>%
  # Group by pitcher, game to isolate unique appearances
  group_by(pitcher, game_pk) %>%
  arrange(game_pk, pitch_number_appearance) %>%
  # Calculate the change in outs between consecutive rows
  mutate(out_diff = c(0,diff(outs_when_up))) %>%
  # Correct for inning transitions by handling negatives in outdiff
  mutate(out_diff = ifelse(out_diff == -1, 2, 
                           ifelse(out_diff==-2,1,out_diff))) %>% 
  # Check for outs in last pitch of appearance
  mutate(out_diff=
           ifelse(pitch_number_appearance!=max(pitch_number_appearance),
                  out_diff,
                  ifelse(events %in% single_outs,1,
                         ifelse(events %in% double_outs,2,
                                ifelse(events == "triple_play",3,
                                       out_diff))))) %>% 
  # Now group by pitcher and role key (SP/RP)
  ungroup() %>% 
  group_by(pitcher,role_key) %>%
  # Calculate stats 
  summarize(
    IP = sum(out_diff, na.rm = TRUE) / 3,
    WHIP = sum(events %in% c("walk", "single", "double", "home_run", 
                             "triple")) / 
      (sum(out_diff, na.rm = TRUE) / 3),
    H = sum(events %in% c("single", "double", "home_run", "triple")),
    HR = sum(events == "home_run"),
    ER = sum(post_bat_score - bat_score),
    SO = sum(events %in% c("strikeout", "strikeout_double_play")),
    BB = sum(events == "walk"),
    HBP = sum(events == "hit_by_pitch"),
    # Per 9 innings stats
    ERA = (sum(post_bat_score - bat_score) / 
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    H9 = (sum(events %in% c("single", "double", "home_run", "triple")) / 
            (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    HR9 = (sum(events == "home_run") / 
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    BB9 = (sum(events == "walk") / 
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    SO9 = (sum(events %in% c("strikeout", "strikeout_double_play")) /
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    # SOs/BBs
    SOBBratio = sum(events %in% c("strikeout", "strikeout_double_play")) /
      sum(events == "walk")
  )
# 2023 stats 
pitcher_stats_23 <- savant_2023 %>%
  # Group by pitcher, game to isolate unique appearances
  group_by(pitcher, game_pk) %>%
  arrange(game_pk, pitch_number_appearance) %>%
  # Calculate the change in outs between consecutive rows
  mutate(out_diff = c(0,diff(outs_when_up))) %>%
  # Correct for inning transitions by handling negatives in outdiff
  mutate(out_diff = ifelse(out_diff == -1, 2, 
                           ifelse(out_diff==-2,1,out_diff))) %>% 
  # Check for outs in last pitch of appearance
  mutate(out_diff=
           ifelse(pitch_number_appearance!=max(pitch_number_appearance),
                  out_diff,
                  ifelse(events %in% single_outs,1,
                         ifelse(events %in% double_outs,2,
                                ifelse(events == "triple_play",3,
                                       out_diff))))) %>% 
  # Now group by pitcher and role key (SP/RP)
  ungroup() %>% 
  group_by(pitcher,role_key) %>%
  # Calculate stats 
  summarize(
    IP = sum(out_diff, na.rm = TRUE) / 3,
    WHIP = sum(events %in% c("walk", "single", "double", "home_run", 
                             "triple")) / 
      (sum(out_diff, na.rm = TRUE) / 3),
    H = sum(events %in% c("single", "double", "home_run", "triple")),
    HR = sum(events == "home_run"),
    ER = sum(post_bat_score - bat_score),
    SO = sum(events %in% c("strikeout", "strikeout_double_play")),
    BB = sum(events == "walk"),
    HBP = sum(events == "hit_by_pitch"),
    # Per 9 innings stats
    ERA = (sum(post_bat_score - bat_score) / 
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    H9 = (sum(events %in% c("single", "double", "home_run", "triple")) / 
            (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    HR9 = (sum(events == "home_run") / 
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    BB9 = (sum(events == "walk") / 
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    SO9 = (sum(events %in% c("strikeout", "strikeout_double_play")) /
             (sum(out_diff, na.rm = TRUE) / 3)) * 9,
    # SOs/BBs
    SOBBratio = sum(events %in% c("strikeout", "strikeout_double_play")) /
      sum(events == "walk")
  )
# Add year to stats
pitcher_stats_21$Year = rep("2021",nrow(pitcher_stats_21))
pitcher_stats_22$Year = rep("2022",nrow(pitcher_stats_22))
pitcher_stats_23$Year = rep("2023",nrow(pitcher_stats_23))
# Combine years and join with batters faced
pitcher_stats <- rbind(pitcher_stats_21 %>%
                         left_join(batters_faced_21,by=c("pitcher","role_key")),
                       pitcher_stats_22 %>%
                         left_join(batters_faced_22,by=c("pitcher","role_key")),                             pitcher_stats_23 %>%
                         left_join(batters_faced_23,by=c("pitcher","role_key")))

# Combine stats created on pitch-type level
pitcher_stats_type <- rbind(pitcher_21,pitcher_22,pitcher_23)
# Pitcher stats to csv
#setwd("/Users/landondocherty/Downloads")
#write.csv(pitcher_stats,"Pitcher_Stats_Full.csv")
#write.csv(pitcher_stats_type,"Pitcher_Stats_Type.csv")

#XGBoost for Relievers
# Filter for only releivers
releivers <- pitcher_stats %>% ungroup() %>% filter(role_key=="RP")
# Deal with inf/na issue for SOBBratio
# When inf, pitcher had 0 walks, but at least 1 strikeout, so make ratio the
# maximum value in data
releivers$SOBBratio <- ifelse(releivers$SOBBratio=="Inf",21,releivers$SOBBratio)
# When na, pitcher had 0 walks and 0 strikeouts, so make ratio 0
releivers$SOBBratio <- ifelse(is.na(releivers$SOBBratio),0,releivers$SOBBratio)
# Deal with inf/na issue with WHIP
# When na, pitcher had no hits, walks, or innings pitched, so make 0
releivers$WHIP <- ifelse(is.na(releivers$WHIP),0,releivers$WHIP)
# When inf, pitcher gave up hits or walks, but did not record an out, so make
# maximum from data
releivers$WHIP <- ifelse(releivers$WHIP=="Inf",12,releivers$WHIP)
# Group into training (2021,2022) and testing (2023)
releivers_train <- releivers %>%
  filter(Year<=2022) %>%
  select(WHIP, H,	HR,	ER,	SO,	BB,	HBP,	SOBBratio, batters_faced)
relievers_test <- releivers %>%
  filter(Year==2023) %>%
  select(WHIP, H,	HR,	ER,	SO,	BB,	HBP,	SOBBratio, batters_faced)
releivers_full <- releivers %>%
  select(WHIP, H,	HR,	ER,	SO,	BB,	HBP,	SOBBratio, batters_faced)
# XGBoost With cross-validation
# Define training matrices
train_x <- model.matrix(batters_faced ~ ., data = releivers_train)[,-1]
train_y <- releivers_train$batters_faced
set.seed(4321)
xgb.ins <- xgb.cv(data = train_x, label = train_y, subsample = 0.5, nrounds = 100, nfold=10,
                  params=list(objective="reg:squarederror"))
# Best test RMSE comes at round 60
# Tune with 60 rounds
# Define tuning grid
tunegrid <- expand.grid(nrounds=60,eta = c(0.1, 0.15, 0.2, 0.25, 0.3),
                        max_depth = c(1:10),gamma = c(0), colsample_bytree = 1,
                        min_child_weight = 1,subsample = c(0.25, 0.5, 0.75, 1))
# Tune for best hyperparameters
xgb.ins.caret <- train(x = train_x, y = as.factor(train_y),
                       method = "xgbTree",
                       tuneGrid = tunegrid,
                       trControl = trainControl(method = 'cv', number = 10))
# Plot and identify best parameters
plot(xgb.ins.caret)
xgb.ins.caret$bestTune
# Fit with best parameters
xgb.relievers <- xgboost(data=train_x,label=train_y,subsample=1,nrounds=60,eta=0.2,
                         maxdepth=10,gamma=0,colsample_bytree=1,min_child_weight=1,
                         params=list(objective="reg:squarederror"))
# Variable importance
xgb.importance(feature_names = colnames(train_x), model = xgb.relievers)
xgb.ggplot.importance(xgb.importance(feature_names = colnames(train_x), model = xgb.relievers))
# Prediction for testing
predict_x <- model.matrix(batters_faced ~.,data=relievers_test)[,-1]
relievers_test$prediction <- as.numeric(as.vector(predict(xgb.relievers, predict_x,type = "response")))
# MAE on testing
mean(abs(relievers_test$batters_faced-relievers_test$prediction))
# MAPE on testing
100 * mean(abs(relievers_test$batters_faced-relievers_test$prediction)/
             relievers_test$batters_faced)
# RMSE on testing
sqrt(mean((relievers_test$batters_faced-relievers_test$prediction)^2))

# 2024 Predictions for Relievers
# X and Y
x <- model.matrix(batters_faced ~ ., data = releivers_full)[,-1]
y <- releivers_full$batters_faced
# Fit with best parameters using all three years
xgb.relievers_full <- xgboost(data=x,label=y,subsample=1,nrounds=60,eta=0.2,
                              maxdepth=10,gamma=0,colsample_bytree=1,min_child_weight=1,
                              params=list(objective="reg:squarederror"))
# Read in projected reliever stats for 2024
setwd("/Users/landondocherty/Downloads")
RP_future <- read.csv("RP_final.csv")
# Drop NA rows
RP_future_no_NA <- RP_future %>% filter(!(is.na(H))) %>% select(-c(X))
# Predict
predict_x <- model.matrix(~.,data=RP_future_no_NA[,-1])[,-1]
RP_future_no_NA$BF_prediction <- as.numeric(as.vector(predict(xgb.relievers_full, predict_x,type = "response")))
# Write Predictions
RP_24 <- RP_future_no_NA %>% select(pitcher,BF_prediction)
write.csv(RP_24,"RP_2024_projections.csv")

#Model for Starters
sp_data <- pitcher_stats %>% ungroup() %>% filter(role_key=="SP")
#Define Features & Target
X_vars <- c("WHIP", "H", "HR", "ER", "SO", "BB", "HBP", "SOBBratio")
y_var <- "batters_faced"
#Split data into training (2021 & 2022) and testing (2023)
set.seed(88)
train_set <- sp_data %>% filter(Year %in% c(2021, 2022))
test_set <- sp_data %>% filter(Year == 2023)
#Extract features & target for train and test sets
X_train <- as.matrix(train_set[, ..X_vars])
y_train <- train_set[[y_var]]
X_test <- as.matrix(test_set[, ..X_vars])
y_test <- test_set[[y_var]]
#Check & Remove NA/Inf values in training and testing sets
valid_rows_train <- complete.cases(X_train) & !apply(X_train, 1, function(row) any(is.infinite(row)))
X_train <- X_train[valid_rows_train, ]
y_train <- y_train[valid_rows_train]
valid_rows_test <- complete.cases(X_test) & !apply(X_test, 1, function(row) any(is.infinite(row)))
X_test <- X_test[valid_rows_test, ]
y_test <- y_test[valid_rows_test]
#### XGBoost and Random Forest Model ####
#Define XGBoost tuning grid
xgb_grid <- expand.grid(
  nrounds = c(200, 400, 600), #Number of boosting rounds
  max_depth = c(3, 6, 9, 12), #Tree depth
  eta = c(0.01, 0.05, 0.1, 0.2), #Learning rate
  gamma = c(0, 1, 5), #Minimum loss reduction required to make further splits
  colsample_bytree = c(0.6, 0.8, 1.0), #Feature sampling
  min_child_weight = c(1, 3, 5), #Minimum sum of weights in a child
  subsample = c(0.6, 0.8, 1.0) #Row sampling
)
#Enable parallel processing
cl <- makeCluster(detectCores() - 1) #Use 7 out of 8 threads
registerDoParallel(cl)
#Train the XGBoost model using cross-validation
train_control <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
xgb_train <- train(
  X_train, y_train,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid
)
#Stop parallel processing
stopCluster(cl)
#Get the best hyperparameters
best_params <- xgb_train$bestTune
print("Best Parameters Selected:")
print(best_params)
#Train the final XGBoost model
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)
final_xgb <- xgb.train(
  params = list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = best_params$max_depth,
    eta = best_params$eta,
    subsample = best_params$subsample,
    colsample_bytree = best_params$colsample_bytree,
    min_child_weight = best_params$min_child_weight
  ),
  data = dtrain,
  nrounds = best_params$nrounds,
  early_stopping_rounds = 50,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 1
)
#Feature Importance
importance <- xgb.importance(feature_names = X_vars, model = final_xgb)
print("Feature Importance:")
print(importance)
#Train Random Forest model
rf_model <- randomForest(X_train, y_train, ntree = 500)
#Make Predictions
y_pred_xgb <- predict(final_xgb, newdata = dtest)
y_pred_rf <- predict(rf_model, newdata = X_test)
#Evaluate Performance
rmse_xgb <- sqrt(mean((y_test - y_pred_xgb)^2))
rmse_rf <- sqrt(mean((y_test - y_pred_rf)^2))
mape_xgb <- mean(abs((y_test - y_pred_xgb) / (y_test + 1e-10))) * 100
mape_rf <- mean(abs((y_test - y_pred_rf) / (y_test + 1e-10))) * 100
mae_xgb <- mean(abs(y_test - y_pred_xgb))
mae_rf <- mean(abs(y_test - y_pred_rf))
#Print Results
cat("\nFinal Model Results:\n")
cat("XGBoost RMSE:", rmse_xgb, " | MAE:", mae_xgb, " | MAPE:", mape_xgb, "\n")
cat("Random Forest RMSE:", rmse_rf, " | MAE:", mae_rf, " | MAPE:", mape_rf, "\n")
#Select the Best Model
if (rmse_xgb < rmse_rf) {
  cat("\nBest Model: XGBoost (Lower RMSE)\n")
  best_model <- final_xgb
  y_pred_best <- y_pred_xgb
} else {
  cat("\nBest Model: Random Forest (Lower RMSE)\n")
  best_model <- rf_model
  y_pred_best <- y_pred_rf
}
#### Predict Batters Faced for 2024 ####
#Load 2024 features dataset
features <- read.csv('C:/Users/ambue/OneDrive/Documents/MSA_25/Reds_Hackathon/SP_final.csv')
#Define the feature columns used in training
X_vars <- c("WHIP", "H", "HR", "ER", "SO", "BB", "HBP", "SOBBratio")
#Exclude the pitcher column from the model features
X_2024 <- features[, X_vars]
#Convert to matrix for XGBoost
X_2024_matrix <- as.matrix(X_2024)
#Handle missing or infinite values
X_2024_matrix[is.na(X_2024_matrix)] <- 0
X_2024_matrix[is.infinite(X_2024_matrix)] <- 0
#Make predictions using the trained model
y_pred_2024 <- predict(best_model, newdata = X_2024_matrix)
#Create a DataFrame with predictions
predictions_2024_df <- data.frame(
  Pitcher_ID = features$pitcher,
  Predicted_Batters_Faced = y_pred_2024
)
#Save the predictions to CSV
write.csv(predictions_2024_df, "C:/Users/ambue/OneDrive/Documents/MSA_25/Reds_Hackathon/2024_predictions_year_split.csv", row.names = FALSE)
print("2024 Predictions saved to 2024_predictions.csv")
3:46
##########

# Complete Pitcher Predictions
# Read in starter projections
setwd("/Users/landondocherty/Downloads")
SP_future <- read.csv("2024_predictions_year_split.csv")
# Define similar column names
colnames(SP_future) <- c("PLAYER_ID","PLAYING_TIME")
colnames(RP_24) <- c("PLAYER_ID","PLAYING_TIME")
# Combine
Pitchers_Future <- rbind(RP_24,SP_future)
# Group by pitcher to sum playing time for pitchers appearing as both starters
# and relievers
Pitchers_Future <- Pitchers_Future %>% group_by(PLAYER_ID) %>%
  summarize(PLAYING_TIME=sum(PLAYING_TIME))
# Write predictions
write.csv(Pitchers_Future,"Pitchers_2024_projections.csv")

# Full projections file
# Read in batter projections
setwd("/Users/landondocherty/Downloads")
batters <- read.csv("Batting_Stats_Final2.csv")
# Define similar column names
colnames(batters) <- c("PLAYER_ID","PLAYING_TIME")
# Join with pitchers
Projections_2024_Full <- rbind(batters,Pitchers_Future)
# Group and summarize to account for players appearing in both sets
Projections_2024_Full <- Projections_2024_Full %>% group_by(PLAYER_ID) %>%
  summarize(PLAYING_TIME=sum(PLAYING_TIME))
# Write to CSV
write.csv(Projections_2024_Full,"2024_Projections_Full.csv")

#Reading in projection data
full_projections <- read.csv("C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\sub\\2024_Projections_Full.csv")
#Reading in sample submission
sample <-read.csv("C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\sub\\sample_submission.csv")
#Dropping sample submission playing time estimate
sample <- sample %>% select(PLAYER_ID)
#Left merge so that we only pull in rows from the sample submission, as per request
submission_pre <- left_join(sample,full_projections, by = "PLAYER_ID" )
#Dealing with NA's by setting them to 0 (players we chose to not project for due to previously missing seasons
submission <- submission_pre %>% mutate(PLAYING_TIME = ifelse(is.na(PLAYING_TIME), 0, PLAYING_TIME))
#Outputting to csv
write.csv(submission, "C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\sub\\submission.csv",row.names = FALSE)