DROP TABLE sns.user_message;
-- Task 8
-- CREATE TABLE user_message 
CREATE TABLE `sns`.`user_message` (
  `id` BIGINT NOT NULL UNIQUE AUTO_INCREMENT,
  `sourceId` BIGINT NOT NULL,
  `targetId` BIGINT NOT NULL,
  `message` TEXT NULL,
  `createAt` DATETIME NOT NULL DEFAULT now(), 
  `updatedAt` DATETIME NULL ,
  PRIMARY KEY(`id`), 
  FOREIGN KEY(`sourceId`) REFERENCES user(`id`), 
  FOREIGN KEY(`targetId`) REFERENCES user(`id`)
);
-- Task 10
-- Insert data
INSERT INTO sns.user_message
(sourceId, targetId, message)
VALUES (1,2,"1 communicate to 2");

INSERT INTO sns.user_message
(sourceId, targetId, message)
VALUES (2,1,"2 communicate to 1");

-- Task 11
-- ALTER ADD COLUMN
ALTER TABLE sns.user_message
	ADD COLUMN status ENUM ('seen','delivered') NULL;
    
SELECT * from sns.user;
SELECT * from sns.user_post;
SELECT * from sns.user_message;