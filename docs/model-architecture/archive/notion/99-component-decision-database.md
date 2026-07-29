<database url="{{https://app.notion.com/p/250be554eacc40219065073dfcf66fd7}}" inline="false">
The title of this Database is: 组件决策记录
<ancestor-path>
<parent-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
Here are the Database's Data Sources:
You can use the "view" tool on the URL of any Data Source to see its full schema configuration.
<data-sources>
<data-source url="{{collection://69ca66ff-43e7-4128-bb7b-9f3751506705}}">
The title of this Data Source is: 组件决策记录

Here is the database's configurable state:
Properties with `readOnly: true` are synced or system-managed. Do not try to update their values with page update tools.
<data-source-state>
{"name":"组件决策记录","schema":{"决定日期":{"description":"","name":"决定日期","querySqlColumns":{"columns":[{"name":"date:决定日期:start","sqlType":"TEXT"},{"name":"date:决定日期:end","sqlType":"TEXT"},{"name":"date:决定日期:is_datetime","sqlType":"INTEGER"}],"usage":"For connections.notion.querySql. Main schema name not queryable."},"type":"date"},"决策编号":{"description":"","name":"决策编号","type":"auto_increment_id"},"序号":{"description":"","name":"序号","type":"number"},"影响":{"description":"","name":"影响","options":[{"color":"red","description":"","name":"高","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/c0k8Qw/YzQzMzEzOTMtNjc4OS00ZDE3LWExMzEtNjlmZWE3YTI3Mzhl"},{"color":"yellow","description":"","name":"中","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/c0k8Qw/MTIzZWQxZGMtMzUzNy00NzczLTk2NWUtMDlkODEzYTQ2NzVi"},{"color":"gray","description":"","name":"低","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/c0k8Qw/ZDlhYzVmODQtYTI1ZS00YjExLTkwNDAtNjA5OWQ3OWU4NzZi"}],"type":"select"},"标签":{"description":"","name":"标签","options":[{"color":"blue","description":"","name":"架构","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/WFpWWQ/YzJhN2VmNGUtOTZkMS00ZTM2LTg2NWUtNzM2ZjIyM2MyYWUz"},{"color":"green","description":"","name":"训练","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/WFpWWQ/MjYzZTMyYzMtZTllOC00YTBkLTg0ZmUtNGVmMWM0OTYxMTlh"},{"color":"orange","description":"","name":"数据","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/WFpWWQ/ZjBjYzU4MWYtNzI5ZC00Y2E0LTg4MmMtMGMyYzIyNDUyOTk1"},{"color":"gray","description":"","name":"系统","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/WFpWWQ/NTBmYzlkMTAtMTBiMC00NGFiLTlhNDgtZTJmYjAzMGUzYzE1"}],"type":"multi_select"},"状态":{"description":"","name":"状态","options":[{"color":"gray","description":"","name":"待讨论","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/YmZEWw/MmFmY2I1YmEtZWY1NS00NDBkLWJlYjAtYzcxODMwNDBjNTY4"},{"color":"blue","description":"","name":"讨论中","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/YmZEWw/NGIyZjNmNzQtYWU4My00NWM1LTgyMTQtYTBiMjk5ZmU1YTMw"},{"color":"yellow","description":"","name":"待验证","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/YmZEWw/N2IyYWI2NWMtYmRjMy00NDE3LTgyMjctMDk5YWE3Y2MzZWVl"},{"color":"green","description":"","name":"已接受","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/YmZEWw/OWU0OWI5NTctOGFmMy00ZmQxLTg0MmItODBlMzQwNjVjMzk1"},{"color":"red","description":"","name":"已否决","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/YmZEWw/MDM3NTk1YzQtZTA3Mi00ZDdmLWEwZmEtNDRmMzAxZTI3ZWEw"},{"color":"purple","description":"","name":"已取代","url":"collectionPropertyOption://69ca66ff-43e7-4128-bb7b-9f3751506705/YmZEWw/ZWFhYjNjMTMtOWQ4Zi00ZTQxLTgwODEtMTZiYmQ0MDdlYmU1"}],"type":"select"},"组件决策":{"description":"","name":"组件决策","type":"title"}},"url":"collection://69ca66ff-43e7-4128-bb7b-9f3751506705"}
</data-source-state>

Here is the SQLite table definition for this data source.
<sqlite-table>
CREATE TABLE IF NOT EXISTS "collection://69ca66ff-43e7-4128-bb7b-9f3751506705" (
	url TEXT UNIQUE,
	createdTime TEXT, -- ISO-8601 datetime string, automatically set. This is the canonical time for when the page was created.
	"date:决定日期:start" TEXT, -- ISO-8601 date or datetime string. Use the expanded property (date:<column_name>:start) to set this value.
	"date:决定日期:end" TEXT, -- ISO-8601 date or datetime string, can be empty. Must be NULL if the date is a single date, and must be present if the date is a range. Use the expanded property (date:<column_name>:end) to set this value.
	"date:决定日期:is_datetime" INTEGER, -- 1 if the date is a datetime, 0 if it is a date, NULL defaults to 0. Use the expanded property (date:<column_name>:is_datetime) to set this value.
	"标签" TEXT, -- JSON array with zero or more of ["架构", "训练", "数据", "系统"]
	"序号" FLOAT,
	"状态" TEXT, -- one of ["待讨论", "讨论中", "待验证", "已接受", "已否决", "已取代"]
	"决策编号" INTEGER,
	"影响" TEXT, -- one of ["高", "中", "低"]
	"组件决策" TEXT
)
</sqlite-table>
</data-source>
</data-sources>
Here are the Database's Views:
You can use the "view" tool on the URL of any View to see its full configuration.
<views>
<view url="{{view://58499e1d-8593-4582-b860-756085e81208}}">
{"dataSourceUrl":"{{collection://69ca66ff-43e7-4128-bb7b-9f3751506705}}","displayProperties":["组件决策","决定日期","决策编号","序号","影响","标签","状态"],"name":"Default view","type":"table"}
</view>
</views>
</database>
