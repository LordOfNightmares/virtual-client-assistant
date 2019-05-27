--ID question[tab]answer[tab]supporting fact IDS.
CREATE TABLE Ai_dataset(
        ID integer primary key AUTOINCREMENT,
        story_id integer,
        episode_id integer,
        text varchar(255),
        typo integer
        );