pyinstaller -y -D video_service.py ^
-p .\interface\python\interface.py ^
-p .\modules\deploy_handle.py ^
-p .\modules\__init__.py ^
-p .\*.py

XCOPY .\models .\dist\video_service\models /E /Y /I
XCOPY .\videoconfig.json .\dist\video_service\