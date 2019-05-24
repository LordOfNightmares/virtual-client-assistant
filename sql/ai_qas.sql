--ID question[tab]answer[tab]supporting fact IDS.
CREATE TABLE Ai_qas(
        ID integer primary key AUTOINCREMENT,
        question varchar(200),--context
        answer varchar(200),--context
        story_id integer,
        facts_id varchar(200)--context
        );