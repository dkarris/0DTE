WITH CallTable AS (
SELECT DATE(Time_Converted) AS 'Date', MAX(options_pct_change) AS 'Call_Max_Daily' FROM data 
WHERE TYPE = 'call'
GROUP BY DATE(Time_Converted)
),
PutTable AS (
SELECT DATE(Time_Converted) AS 'Date', MAX(options_pct_change) AS 'Put_Max_Daily' FROM data 
WHERE TYPE = 'put'
GROUP BY DATE(Time_Converted)
)
SELECT CallTable.*, PutTable.Put_Max_Daily FROM CallTable 
JOIN PutTable ON CallTable.Date = PutTable.Date


SELECT DATE(Time_Converted) AS 'Date', MAX(options_pct_change) AS 'Call_Max_Daily' FROM data 
WHERE TYPE = 'call'
GROUP BY DATE(Time_Converted)

SELECT
  DATE(Time_Converted) AS 'Date',
  MAX(CASE WHEN TYPE = 'call' THEN options_pct_change ELSE NULL END) AS 'Call_Max_Daily',
  MAX(CASE WHEN TYPE = 'put' THEN options_pct_change ELSE NULL END) AS 'Put_Max_Daily'
FROM data
WHERE DTE_ADJUSTED = 0 AND options_earliest_open>0.2
GROUP BY DATE(Time_Converted)



SELECT * FROM data WHERE DATE([time_converted]) = '2023-03-09' 
AND TYPE = 'put' AND dte_adjusted = 0


  WITH 
        
        MAX_TABLE AS (      
        select distinct date(time_converted) as date
        ,day_name
        , DTE_adjusted
        , day_classification
        , max(options_pct_change) as max_options_pct_change
        , time_converted as time_of_max_change
         from data
         where options_earliest_open>0.15
        group by date(time_converted), dte
        having max_options_pct_change = options_pct_change
        ), 
        
        MIN_TABLE AS (
        select distinct date(time_converted) as date
        , DTE_adjusted
        , day_classification
        , min(options_pct_change) as min_options_pct_change
        , time_converted as time_of_min_change
         from data
         where options_earliest_open>0.15
        group by date(time_converted), dte
        having min_options_pct_change = options_pct_change
        )
        
        select 
        MAX_TABLE.date, MAX_TABLE.day_name, MAX_TABLE.DTE_adjusted, 
        MAX_TABLE.day_classification,MAX_TABLE.max_options_pct_change, MAX_TABLE.time_of_max_change
        , MIN_TABLE.min_options_pct_change, MIN_TABLE.time_of_min_change 
        from MAX_TABLE
        LEFT JOIN MIN_TABLE ON
        MAX_TABLE.date = MIN_TABLE.date 
        AND MAX_TABLE.DTE_adjusted = MIN_TABLE.DTE_adjusted
