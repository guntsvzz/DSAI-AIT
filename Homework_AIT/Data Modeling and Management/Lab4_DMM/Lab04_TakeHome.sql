-- Task1 Show all group information and the creator of the group
use sns;
select g.title, g.details, g.createdBy, u.firstName AS targetFirstName, u.lastName AS targetLastName
from sns.user u inner join sns.group g 
on u.id = g.createdBy;

-- Task2 Suppose there is a user (id= 55), who needs to know who has rejected their friend request. 
-- Show all names and their contact information (mobile and email). 
-- Answer using both JOIN keyword and subquery (total of 2 commands for this task). 
USE sns;
SELECT CONCAT(u.firstName,' ',u.lastName) AS users, u.mobile, u.email
FROM sns.user u 
INNER JOIN sns.user_friend uf ON u.id = uf.targetId 
WHERE u.id IN (SELECT targetId FROM sns.user_friend uf WHERE uf.sourceId = 55 AND uf.status = 'rejected');

-- Task3 Show the first approved post of each group with the full name of the group, 
-- full name of the poster and the date the post was created. 
USE sns;
SELECT g.id, g.title, 
(SELECT gp.message FROM group_post gp 
	WHERE gp.groupId = g.id ORDER BY createdAt LIMIT 1) AS first_message , 
(SELECT gp.createdAt FROM group_post gp 
	WHERE gp.groupId = g.id ORDER BY createdAt LIMIT 1) AS post_date
FROM sns.group g;

-- Task4 Summarize how many posts are created on each group wall. 
-- Separate columns by group name and the total number of the posters for each gender. 
-- Hint: MalePosts | FemalePosts | UnidentifiedPosts
USE sns;
SELECT g.title ,
(SELECT COUNT(gm.id) FROM group_member gm 
	INNER JOIN sns.user u ON u.id = gm.userId 
    WHERE gm.groupId = g.id AND u.gender = 'm') 
    AS MalePost,
(SELECT COUNT(gm.id) FROM group_member gm 
	INNER JOIN sns.user u ON u.id = gm.userId 
    WHERE gm.groupId = g.id AND u.gender = 'f') 
    AS FemalePost,
(SELECT COUNT(gm.id) FROM group_member gm 
	INNER JOIN sns.user u ON u.id = gm.userId 
    WHERE gm.groupId = g.id AND u.gender IS NULL ) 
    AS UnidentifiedPosts
FROM sns.group g;

-- Task5 Show all users who have had no interactions; 
-- not receive and friend request and sent no friend requests. 
-- Display their names and email addresses. 
USE sns;
SELECT u.id, CONCAT(u.firstName,' ',u.lastName) AS USERS, u.email 
FROM sns.user u 
WHERE u.id NOT IN (SELECT sourceId FROM user_friend)  -- no receive
AND u.id NOT IN (SELECT targetId FROM user_friend); -- no friend requests

-- Task6 Show the inactive members (members with no posts) of the group Id 15. 
-- List the full name, user id along with the name of the group.
USE sns;
SELECT DISTINCT gm.userId FROM group_member gm 
INNER JOIN group_post gp ON gp.groupId = gm.groupId
WHERE gm.groupId = 15 AND 
gm.userId NOT IN (SELECT userId FROM group_post WHERE groupId = 15) 
AND gm.status = 'approved';

-- Task7 Show all users who have their message wall empty. 
USE sns; 
SELECT * FROM user u  
WHERE u.id NOT IN 
(SELECT wallId FROM user_post WHERE wallId IS NOT NULL) ;

SELECT wallId FROM user_post WHERE wallId IS NOT NULL;

-- Task8 Show the user information with the number of posts created by each user 
-- (both on friends, public, and group wall). 
USE sns;
SELECT u.id, u.firstName, u.email ,
(SELECT COUNT(up.id) FROM user_post up WHERE up.posterId = u.id AND up.wallId IS NULL) AS PublicPosts,
(SELECT COUNT(up.id) FROM user_post up WHERE up.posterId = u.id AND up.wallId IS NOT NULL) AS PostsToFriend,
(SELECT COUNT(gp.id) FROM group_post gp WHERE gp.userId = u.id ) AS GroupPosts,
((SELECT COUNT(up.id) FROM user_post up WHERE up.posterId = u.id AND up.wallId IS NULL)
+(SELECT COUNT(up.id) FROM user_post up WHERE up.posterId = u.id AND up.wallId IS NOT NULL)
+(SELECT COUNT(gp.id) FROM group_post gp WHERE gp.userId = u.id )) AS TotalPosts
FROM user u;

-- Task9 Summarize the group information by the number of members. 
USE sns;
SELECT 
(SELECT COUNT(a.id) FROM 
	(SELECT g.id,COUNT(gm.userId) AS members FROM group_member gm 
		INNER JOIN `group` g ON gm.groupId = g.id WHERE gm.status = 'approved' GROUP BY g.id ORDER BY g.id) 
    a WHERE a.members < 5) AS LightGroup ,
(SELECT COUNT(a.id) FROM 
	(SELECT g.id,COUNT(gm.userId) AS members FROM group_member gm 
		INNER JOIN `group` g ON gm.groupId = g.id WHERE gm.status = 'approved' GROUP BY g.id ORDER BY g.id) 
	a WHERE a.members > 5 AND a.members <10) AS MediumGroup ,
(SELECT COUNT(a.id) FROM 
	(SELECT g.id,COUNT(gm.userId) AS members FROM group_member gm 
		INNER JOIN `group` g ON gm.groupId = g.id WHERE gm.status = 'approved' GROUP BY g.id ORDER BY g.id) 
	a WHERE a.members > 10) AS DenseGroup ;

-- Task10 Summarize the user table by the range of membership duration (in DAYs). Display the following columns. 
SELECT DISTINCT
(SELECT COUNT(a.id) FROM (SELECT u.id,timestampdiff(DAY,u.registeredAt,now()) AS days FROM user u) a WHERE days<14) AS Newbie ,
(SELECT COUNT(a.id) FROM (SELECT u.id,timestampdiff(DAY,u.registeredAt,now()) AS days FROM user u) a WHERE days BETWEEN 14 and 20) AS Pros,
(SELECT COUNT(a.id) FROM (SELECT u.id,timestampdiff(DAY,u.registeredAt,now()) AS days FROM user u) a WHERE days>20) AS Veterans
From sns.user;