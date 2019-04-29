CREATE TABLE Messages(
        ID integer primary key AUTOINCREMENT,
        title varchar(100), --
        body varchar(2000),
       	user_id integer,
       	bot_id integer,
       	conversation_id varchar(100),
        created timestamp,
        modified timestamp,
        accessed timestamp 
        --to see if the message was seen
        );