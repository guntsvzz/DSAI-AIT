DROP TABLE sns.user_post;
-- Task 1
-- CREATE TABLE user_post 
CREATE TABLE `sns`.`user_post` (
  `id` BIGINT NOT NULL UNIQUE AUTO_INCREMENT,
  `posterId` BIGINT NOT NULL, 
  `profileId` BIGINT NULL,
  `message` TEXT NULL,
  `createAt` DATETIME NOT NULL DEFAULT now(), 
  `updatedAt` DATETIME NULL ,
  PRIMARY KEY(`id`), 
  FOREIGN KEY(`posterId`) REFERENCES user(`id`) ON DELETE CASCADE,
  FOREIGN KEY(`profileId`) REFERENCES user(`id`) ON DELETE CASCADE
);

-- Adding constraints (Foreign Keys)
-- ALTER TABLE sns.user_post
-- 	ADD CONSTRAINT fk_friend_source
-- 		FOREIGN KEY (posterId) REFERENCES sns.user(id) 
-- 		ON DELETE CASCADE,
-- 	ADD CONSTRAINT fk_friend_target
-- 		FOREIGN KEY (profileId) REFERENCES sns.user(id) 
-- 		ON DELETE CASCADE,
-- 	ADD UNIQUE uq_friends (sourceId, targetId);
    
-- Task 4
--  Register a new user
-- public post
INSERT INTO sns.user_post
(posterId, message)
VALUES (1,"Hi");

-- other user's wall
INSERT INTO sns.user_post
(posterId, profileId, message)
VALUES (2,1,"Bye");

-- Task5
-- Terminating an account
DELETE FROM sns.user_post WHERE posterId=3;

-- Task6
-- Adding a column 
ALTER TABLE sns.user_post
	ADD COLUMN photos BINARY(1) NULL;

-- Task 7
DELETE FROM sns.user_post WHERE id=2;

SELECT * from sns.user;
SELECT * from sns.user_post;
SELECT * from sns.user_message;