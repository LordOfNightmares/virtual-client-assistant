--ID aspect text
CREATE TABLE Ai_episodes(
        ID integer primary key AUTOINCREMENT,
        story_id integer,
        episode_id integer,
        text varchar(3000)--context
        );