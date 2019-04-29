CREATE TABLE Users(
        ID integer primary key AUTOINCREMENT,
        first_name varchar(10), --
        last_name varchar(10),
        email varchar(20),
        phone varchar(20),
        ai tinyint(1),
        created timestamp,
        modified timestamp,
        accessed timestamp
        --to see if the user is active
        );