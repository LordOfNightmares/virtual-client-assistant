CREATE TABLE Conversations(
        ID varchar(100) PRIMARY KEY,
        mark decimal(3,1),
        feedback varchar(2000),
       	user_id integer,
       	bot_id integer,
        created timestamp,
        modified timestamp,
        ended timestamp
        --to see if the message was seen
        );


