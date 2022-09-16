-- UNION --
use sns;
select * from user_post;
select * from group_post;

SELECT id, posterId, 
'Wall post' AS postType, message
FROM user_post WHERE posterId=21
UNION 
SELECT id, userId, 
'Group post' AS postType, message
FROM group_post WHERE userId=21;

-- INNER JOIN: Get Target Id full name --
Select tg.sourceId, tg.targetId, u.firstName AS targetFirstName, u.lastName as targetLastName
From user u INNER JOIN user_friend tg
ON u.id = tg.targetId;


-- INNER JOIN 2 --
SELECT fr.sourceId, sc.firstName as Source_FirstName, fr.targetId, tg.firstName AS Target_FirstName
FROM user_friend fr INNER JOIN user sc
ON fr.sourceId = sc.id
INNER JOIN user tg
ON fr.targetId = tg.id;

-- LEFT JOIN Example -- 
SELECT gp.groupId, gp.userId, gp.message, gp.status, gp.statusUpdatedBy, u.firstName
FROM group_post gp LEFT JOIN user u
ON u.id = gp.statusUpdatedBy;

-- LEFT JOIN Example 2 -- 
SELECT gp.groupId, g.title, gp.userId, gp.message 
FROM social.group g LEFT JOIN group_post gp
ON g.id = gp.groupId;

-- Full Outter Join -- 
Select CONCAT(u.firstName, ' ', u.lastName) AS USERS, gp.id AS GroupID 
FROM user u LEFT JOIN group_member gp ON u.id = gp.userId
UNION
SELECT CONCAT(u.firstName, ' ', u.lastName) AS USERS, gp.id AS GroupID 
FROM group_member gp RIGHT JOIN user u ON gp.userId = u.id;

-- Sub QUERY --
SELECT u.firstName, u.lastName, u.email 
FROM social.user u
WHERE u.id IN ( SELECT targetId FROM user_friend WHERE sourceId = 99);

-- Sub QUERY: 2 --
SELECT u.Id, CONCAT(u.firstName, ' ', COALESCE(u.middleName, ' '), ' ',   u.lastName) 
	AS PosterName, u.mobile, u.email 
FROM user u
WHERE u.Id IN (
				SELECT userId FROM group_post WHERE groupId = 15 and userId -- user if from the group post table for group 15 and created by -- user 1 to n except 41,42 and 78
                NOT IN ( -- 1 to n except 41,42 and 78
					SELECT userId FROM group_member WHERE groupId = 15
				)
);

SELECT userId FROM group_member WHERE groupId = 15; -- 41,42,48 belong to group 15

-- SUB QUERY: Example 3 -- 
SELECT g.id, g.title,
	(SELECT COUNT(gm.id) FROM group_member gm
		INNER JOIN user u ON gm.userId = u.id
			WHERE gm.groupId = g.id AND u.gender = 'm') AS maleMember,
	(SELECT COUNT(gm.id) FROM group_member gm
		INNER JOIN user u ON gm.userId = u.id
			WHERE gm.groupId = g.id AND u.gender = 'f') AS femaleMember,
	(SELECT COUNT(gm.id) FROM group_member gm
		INNER JOIN user u ON gm.userId = u.id
			WHERE gm.groupId = g.id AND u.gender is NULL) AS unspecifiedMember
FROM sns.group g;

