-- meta=init_sections=create_tables,create_idx


------ DML creates

-- name=create_idx
create index frameset_file on frameset(file);
create index predicate_fs_uid on predicate(fs_uid);
create index roleset_p_uid on roleset(p_uid);
create index roleset_id on roleset(id);
create index roleset_nme on roleset(nme);
create index alias_rs_uid on alias(rs_uid);
create index alias_pos_id on alias(pos_id);
create index alias_word on alias(word);
create index example_rs_uid on example(rs_uid);
create index example_name on example(name);
create index example_source on example(source);
create index example_txt on example(txt);
create index role_rs_uid on role(rs_uid);
create index role_descr on role(descr);
create index role_func_id on role(func_id);
create index role_dx on role(idx);
create index rolelink_rol_uid on rolelink(rol_uid);
create index rolelink_cls on rolelink(cls);
create index rolelink_roleres_id on rolelink(roleres_id);
create index rolelink_version on rolelink(version);
create index rolelink_nme on rolelink(nme);
create index function_uid on function(uid);
create index function_label on function(label);
create index function_descr on function(descr);
create index function_grp on function(grp);
create index pos_uid on pos(uid);
create index pos_nme on pos(nme);
create index roleres_uid on roleres(uid);
create index roleres_nme on roleres(nme);
create index relation_label on relation(label);
create index relation_type on relation(type);
create index relation_reif on relation(reification);

-- name=create_tables
create table frameset (file text);
create table predicate (fs_uid int, lemma text);
create table roleset (p_uid int, id varchar(20), nme text);
create table alias (rs_uid int, pos_id int, word text);
create table example (rs_uid int, name text, source text, txt text, propbank text);
create table role (rs_uid int, descr text, func_id int, idx char(1));
create table rolelink (rol_uid int, cls text, roleres_id int, version varchar(10), nme text);
create table function (uid int primary key unique, label varchar(3) unique, descr text, grp text);
create table pos (uid int primary key unique, nme varchar(20));
create table roleres (uid int primary key unique, nme varchar(20));
create table relation (label text unique, type varchar(15), descr text, regex text, reification text);

------ dbutil hooks

-- name=insert_frameset
insert into frameset (file) values (?);
-- name=select_frameset
select rowid as uid, file from frameset;
-- name=select_frameset_by_id
select rowid as uid, file from frameset where rowid = ?;
-- name=select_frameset_by_file_name
select rowid as uid, file from frameset where file= ?;
-- name=select_frameset_keys
select rowid from frameset;

-- name=insert_predicate
insert into predicate (fs_uid, lemma) values (?, ?);
-- name=select_predicate_by_par_id
select rowid as uid, lemma from predicate where fs_uid = ?;
-- name=select_predicate_by_lemma
select rowid as uid, lemma from predicate where lemma = ?;
-- name=select_predicate_by_id
select rowid as uid, lemma from predicate where rowid = ?;
-- name=select_predicate_keys
select rowid from predicate;
-- name=select_predicate_id_to_uid
select rowid from predicate where lemma = ?;
-- name=select_predicate_ids
select lemma from predicate;

-- name=insert_roleset
insert into roleset (p_uid, id, nme) values (?, ?, ?);
-- name=select_roleset_by_par_id
select rowid as uid, id, nme from roleset where p_uid = ?;
-- name=select_roleset_by_id
select rowid as uid, id, nme from roleset where rowid = ?;
-- name=select_roleset_by_role_id
select rowid as uid, id, nme from roleset where id = ?;
-- name=select_roleset_keys
select rowid from roleset;
-- name=select_roleset_id_to_uid
select rowid from roleset where id = ?;
-- name=select_roleset_ids
select id from roleset;

-- name=insert_alias
insert into alias (rs_uid, pos_id, word) values (?, ?, ?);
-- name=select_alias_by_par_id
select rs_uid, pos_id, word from alias where rs_uid = ?;

-- name=insert_example
insert into example (rs_uid, name, source, txt, propbank) values (?, ?, ?, ?, ?);
-- name=select_example_by_par_id
select rs_uid, name, source, txt, propbank from example where rs_uid = ?;

-- name=insert_role
insert into role (rs_uid, descr, func_id, idx) values (?, ?, ?, ?);
-- name=select_role_by_par_id
select rowid as uid, descr, func_id, idx from role where rs_uid = ?;

-- name=insert_rolelink
insert into rolelink (rol_uid, cls, roleres_id, version, nme) values (?, ?, ?, ?, ?);
-- name=select_rolelink_by_par_id
select rowid as uid, cls, roleres_id, version, nme from rolelink where rol_uid = ?;

-- name=insert_function
insert into function (uid, label, descr, grp) values (?, ?, ?, ?);
-- name=select_function
select uid, label, descr, grp from function;

-- name=insert_pos
insert into pos (uid, nme) values (?, ?);
-- name=insert_roleres
insert into roleres (uid, nme) values (?, ?);

-- name=insert_relation
insert into relation (label, type, descr, regex, reification) values (?, ?, ?, ?, ?);
-- name=select_relation
select rowid, label, type, descr, regex, reification
  from relation
  order by rowid;
-- name=select_relation_id_to_uid
select rowid from relation where label = ?;
-- name=select_relation_ids
select label from relation;
