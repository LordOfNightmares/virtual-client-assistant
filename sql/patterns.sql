--ID question[tab]answer[tab]supporting fact IDS.
CREATE TABLE Patterns(
        ID integer primary key AUTOINCREMENT,
        title varchar(100), --aspect
        body varchar(3000),--context
       	user_id integer,
       	hits integer,--folosit
        misses integer,--ignorat
        created timestamp,
        modified timestamp,
        accessed timestamp
        --to see if the message was seen
        );