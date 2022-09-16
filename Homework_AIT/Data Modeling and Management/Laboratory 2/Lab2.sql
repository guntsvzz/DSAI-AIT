DROP DATABASE sns;
-- CREATE DATABASE
CREATE DATABASE sns; 
-- CREATE TABLE user 
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
  `lastLogin` DATETIME NULL, 
  `intro` TINYTEXT NULL,
  `bio` TEXT NULL, 
  PRIMARY KEY (`id`) -- declare PK
);

-- Adding UNIQUE columns 
ALTER TABLE sns.user
ADD UNIQUE (`username`), 
ADD UNIQUE (`mobile`), 
ADD UNIQUE (`email`);

--  Register a new user
INSERT INTO sns.user
(firstName, middleName, lastName, username, mobile, email, 
passwordHash, registeredAt)
VALUES 
('Sergio', 'Checo', 'Perez', 
'checo11','6415499120','checo.perez@gmail.com',md5('redbull@11'),
now());

-- Exercise 1
-- user 1
INSERT INTO sns.user
(firstName, middleName, lastName, username, mobile, email, 
passwordHash, registeredAt)
VALUES 
('Todsavad', '', 'Tangtortan', 
'guntsv','0846541717','guntsvzz@gmail.com',md5('redbull@11'),
now());
-- user 2
INSERT INTO sns.user
(firstName, middleName, lastName, username, mobile, email, 
passwordHash, registeredAt)
VALUES 
('Passamon', '', 'Teachasit', 
'Passamon','0923728929','passamon@gmail.com',md5('redbull@11'),
now());
-- user 3
INSERT INTO sns.user
(firstName, middleName, lastName, username, mobile, email, 
passwordHash, registeredAt,intro)
VALUES 
('Rapeepat', '', 'Suputtayanggul', 
'Rapeepat ','0620935170','rapeepat@gmail.com',md5('redbull@11'),
now(),'I\'m P 21 years old');
-- user 4
INSERT INTO sns.user
(firstName, middleName, lastName, username, mobile, email, 
passwordHash, registeredAt)
VALUES 
('Saranja   ', '', 'Dunkel', 
'Saranja ','0945454352','saranja@gmail.com',md5('redbull@11'),
now());

-- Adding a column 
ALTER TABLE sns.user
ADD COLUMN gender ENUM ('m','f') NULL;

--  UPDATE user
UPDATE sns.user
SET gender = 'm', 
intro = 'Formula One | Red Bull Racing',
bio = 'Sergio Michel Checo Perez is a mexican born Formula One racing 
driver, driving for Red Bull Racing'
WHERE user.id = 1;

-- CREATE TABLE user_friend 
CREATE TABLE `sns`.`user_friend` (
 `id` BIGINT NOT NULL AUTO_INCREMENT, 
 `sourceId` BIGINT NOT NULL,
 `targetId` BIGINT NOT NULL, 
 `status` ENUM ('new','accepted','rejected') DEFAULT 'new',
 `createdAt` DATETIME NOT NULL,
 `updatedAt` DATETIME NULL,
 PRIMARY KEY (`id`)
);

-- Adding constraints (Foreign Keys)
ALTER TABLE sns.user_friend
	ADD CONSTRAINT fk_friend_source
		FOREIGN KEY (sourceId) REFERENCES sns.user(id) 
		ON DELETE RESTRICT ON UPDATE RESTRICT,
	ADD CONSTRAINT fk_friend_target
		FOREIGN KEY (targetId) REFERENCES sns.user(id) 
		ON DELETE RESTRICT ON UPDATE RESTRICT,
	ADD UNIQUE uq_friends (sourceId, targetId);

-- Send a friend request
INSERT INTO sns.user_friend (sourceId, targetId, createdAt) 
VALUES (1, 5, now());

-- Response to the request
UPDATE sns.user_friend 
SET status = 'accepted'
WHERE sourceId = 1 AND targetId = 5;

-- Exercise 2
-- 1 two friend requests
INSERT INTO sns.user_friend (sourceId, targetId, createdAt) 
VALUES (2, 4, now());
INSERT INTO sns.user_friend (sourceId, targetId, createdAt) 
VALUES (4, 2, now());
-- 2 one for acceptation and another for rejection

-- Terminating an account
-- DELETE FROM sns.user WHERE id=3;

    
-- Changing FK constraints
-- Drop all FKs and indicies first
ALTER TABLE sns.user_friend
	DROP FOREIGN KEY fk_friend_source,
	DROP FOREIGN KEY fk_friend_target,
	DROP INDEX uq_friends;
-- Then recreate FKs and a unique key
ALTER TABLE sns.user_friend
	ADD CONSTRAINT fk_friend_source
		FOREIGN KEY (sourceId) REFERENCES sns.user(id) 
		ON DELETE CASCADE,
	ADD CONSTRAINT fk_friend_target
		FOREIGN KEY (targetId) REFERENCES sns.user(id) 
		ON DELETE CASCADE,
	ADD UNIQUE uq_friends (sourceId, targetId);
    
-- Is able to delete

-- user_postuser_frienduser 
SELECT * from sns.user; -- 1,2,45
SELECT * from sns.user_friend; -- 1,3,4,5