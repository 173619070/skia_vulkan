
wxWidgets 是从 OrcaSlicer 项目的依赖构建 中获取的，而不是单独下载的。

📂 目录结构
1. 根目录
\OrcaSlicer\deps\build\OrcaSlicer_dep\usr\local\
2. 头文件 (Include)
\OrcaSlicer\deps\build\OrcaSlicer_dep\usr\local\include\
├── wx-3.1\          # wxWidgets 3.1 版本头文件
│   └── wx\
│       ├── wx.h
│       ├── app.h
│       ├── frame.h
│       └── ... (所有 wxWidgets 头文件)
└── msvc\            # MSVC 特定配置
    └── wx\
        └── setup.h  # 编译配置
3. 库文件 (Libraries)
\OrcaSlicer\deps\build\OrcaSlicer_dep\usr\local\lib\vc_x64_lib\
├── wxbase31u.lib           # 基础库
├── wxbase31u_net.lib       # 网络库
├── wxbase31u_xml.lib       # XML 库
├── wxmsw31u_core.lib       # 核心 GUI 库 ⭐
├── wxmsw31u_adv.lib        # 高级控件
├── wxmsw31u_aui.lib        # AUI 界面框架
├── wxmsw31u_gl.lib         # OpenGL 支持
├── wxmsw31u_html.lib       # HTML 控件
├── wxmsw31u_media.lib      # 媒体播放
├── wxmsw31u_propgrid.lib   # 属性网格
├── wxmsw31u_ribbon.lib     # Ribbon 界面
├── wxmsw31u_richtext.lib   # 富文本
├── wxmsw31u_qa.lib         # QA 工具
└── wxregexu.lib            # 正则表达式
4. DLL 文件 (运行时)
\OrcaSlicer\deps\build\OrcaSlicer_dep\usr\local\bin\
├── wxbase31u_vc_x64_custom.dll
├── wxmsw31u_core_vc_x64_custom.dll
└── ... (其他 DLL)
🔧 CMake 如何找到 wxWidgets
在 
CMakeLists.txt
 中：

find_package(wxWidgets COMPONENTS core base QUIET)
CMake 通过以下方式找到 wxWidgets：

环境变量或注册表：检查 wxWidgets_ROOT_DIR
CMake 缓存：从之前的配置中读取
自动搜索：在标准路径中查找 wxWidgets-config.cmake
在你的情况下，CMake 找到了：

wxWidgets_ROOT_DIR = /OrcaSlicer/deps/build/OrcaSlicer_dep/usr/local
wxWidgets_LIB_DIR  = /OrcaSlicer/deps/build/OrcaSlicer_dep/usr/local/lib/vc_x64_lib
📦 wxWidgets 版本信息
版本：3.1.x (从库文件名 wxbase31u.lib 可以看出)
配置：
31 = 版本 3.1
u = Unicode 版本
vc_x64 = Visual C++ 64位编译
构建类型：DLL 版本（动态链接库）
🔗 你的程序如何使用
在 wxSkiaVulkanDemo.cpp 中：

#include <wx/wx.h>  // 从 \OrcaSlicer\deps\...\include\wx-3.1\wx\wx.h
编译时链接：

wxmsw31u_core.lib  (核心 GUI 功能)
wxbase31u.lib      (基础功能)
运行时需要：

wxmsw31u_core_vc_x64_custom.dll
wxbase31u_vc_x64_custom.dll
🏗️ OrcaSlicer 如何构建 wxWidgets
OrcaSlicer 使用自己的依赖构建系统：

OrcaSlicer/
├── deps/
│   ├── deps_src/          # 依赖源代码（包括 wxWidgets 源码）
│   └── build/
│       └── OrcaSlicer_dep/
│           └── usr/local/ # 编译后的依赖库 ⭐
构建过程（通常在 OrcaSlicer 的构建脚本中）：

下载 wxWidgets 源码到 deps_src/
使用 CMake 配置和编译
安装到 deps/build/OrcaSlicer_dep/usr/local/
✅ 总结
项目	位置
wxWidgets 根目录	\OrcaSlicer\deps\build\OrcaSlicer_dep\usr\local\
头文件	.../include/wx-3.1/
静态库	.../lib/vc_x64_lib/*.lib
动态库 (DLL)	.../bin/*.dll
版本	wxWidgets 3.1.x Unicode
来源	OrcaSlicer 依赖构建系统
你的 wxSkiaVulkanDemo.cpp 正在使用 OrcaSlicer 已经编译好的 wxWidgets 库，这样可以避免重复编译，并确保版本一致性。
