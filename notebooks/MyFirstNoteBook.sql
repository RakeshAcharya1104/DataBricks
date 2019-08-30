-- Databricks notebook source
  select * from sandbox.finan

-- COMMAND ----------

select * from sandbox.sentimentdata_csv where _c1 is null

-- COMMAND ----------

delete from sandbox.sentimentdata_csv where _c1 is null

-- COMMAND ----------

