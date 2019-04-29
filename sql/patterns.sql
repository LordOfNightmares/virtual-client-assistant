CREATE TABLE Patterns(
        ID integer primary key AUTOINCREMENT,
        title varchar(100), --
        body varchar(2000),
       	user_id integer,
       	hits integer,--folosit
        misses integer,--ignorat
        created timestamp,
        modified timestamp,
        accessed timestamp
        --to see if the message was seen
        );