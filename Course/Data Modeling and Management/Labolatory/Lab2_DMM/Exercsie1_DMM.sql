DROP DATABASE sns;
CREATE DATABASE sns;
CREATE TABLE `sns`.`user` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `firstName` VARCHAR(50) NULL, 
  `middleName` VARCHAR(50) NULL,
  `lastName` VARCHAR(50) NULL, 
  `username` VARCHAR(50) NULL,
  `mobile` VARCHAR(15) NULL, 
  `email` VARCHAR(50) NOT NULL,
  `passwordHash` VARCHAR(32) NOT NULL, 
  `registeredAt` DATETIME NOT NULL DEFAULT now(),
  `lastLogin` DATETIME NULL, `intro` TINYTEXT NULL,
  `bio` TEXT NULL,
  PRIMARY KEY (`id`) 
);
SELECT * from sns.user;
ALTER TABLE sns.user	
	ADD UNIQUE (`username`),
	ADD UNIQUE (`mobile`),    
	ADD UNIQUE (`email`);

-- INSERT INTO sns.user
-- 	(firstName, lastName, middleName, username, mobile, email, passwordHash, registeredAt)
-- VALUES
-- 	('Sergio', 'Checo', 'Perez', 'checo11', '6415499120', 'checo.perez@gmail.com', md5('redbull@11'), now());

INSERT INTO sns.user
	(firstName, lastName, username, mobile, email, passwordHash)
VALUES
	('Todsavad', 'Tangtortan', 'guntsv', '0846541717', 'guntsvzz@gmail.com', md5('gun5555'));
    
-- Exercise 1
-- User 1  
INSERT INTO sns.user
	(firstName, lastName, username, mobile, email, passwordHash)
VALUES
	('Teerapat', 'two', 'three', '0831918123', 'four@gmail.com', md5('five5555'));   
-- User 2 
INSERT INTO sns.user
	(firstName, middleName, lastName, username, mobile, email, passwordHash)
VALUES
	('Kanyasorn', 'S' ,'tee', 'seven', '12747212', 'HI@gmail.com', md5('five5555'));   
-- User 3   
INSERT INTO sns.user
	(firstName, email,passwordHash,bio)
VALUES
	('Passamon','Naenu@gmail',md5('shfuyfgw12in'),'I\'m Passamon Taechasit');      
INSERT INTO sns.user
	(firstName, email,passwordHash)
VALUES
	('Rapeepat','sddsg@gmail',md5('dsds'));    
    
 -- ALTER TABLE column 
ALTER TABLE sns.user
	ADD COLUMN gender ENUM ('m','f') NULL;
   
 -- UPDATE content in the table --
UPDATE sns.user
 SET gender = 'm'
 WHERE user.id = 1;

-- Create new Table User Friend --
CREATE TABLE `sns`.`user_friend` (
  `id` BIGINT NOT NULL AUTO_INCREMENT, 
  `sourceId` BIGINT NOT NULL,
  `targetId` BIGINT NOT NULL, 
  `status` ENUM ('new','accepted','rejected') DEFAULT 'new',
  `createdAt` DATETIME NOT NULL,
  `updatedAt` DATETIME NULL,
  PRIMARY KEY (`id`)
);

-- Add Constraints Foreign Key --
 ALTER TABLE sns.user_friend	
	ADD CONSTRAINT fk_friend_source		
		FOREIGN KEY (sourceId) REFERENCES sns.user(id)        
		ON DELETE RESTRICT ON UPDATE RESTRICT,	
	ADD CONSTRAINT fk_friend_target		
		FOREIGN KEY (targetId) REFERENCES sns.user(id)        
		ON DELETE RESTRICT ON UPDATE RESTRICT,	
	ADD UNIQUE uq_friends (sourceId, targetId);
    
-- Create a friend request
INSERT INTO sns.user_friend (sourceId, targetId, createdAt) 
VALUES (4, 5, now());

-- INSERT INTO sns.user_friend (sourceId, targetId, createdAt) 
-- VALUES (5,6 , now());


-- Accept Friend Request
UPDATE sns.user_friend
SET status = 'accepted'
WHERE sourceId = 4 AND targetId = 5;

-- DELETE FROM sns.user WHERE id = 4;

-- SELECT u.firstName as Sender from sns.user u, sns.user_friend f
-- WHERE u.id = f.sourceId;

-- SELECT * from sns.user_friend; -- 1,2,4,5

SELECT * from sns.user; -- 1,3,4 and 5

-- DELETE FROM sns.user where id = 3;


