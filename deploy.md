1. Flask アプリを GitHub に push
2. Azure App Service を作成
Azure Portal で以下を設定：
- ランタイム：Python 3.x
- OS：Linux
- App Service プラン：B1 など
- デプロイ方法：GitHub Actions
3. GitHub Actions ワークフローを作成
.github/workflows/azure.yml を作成：
```azure.yml
name: Deploy Flask to Azure

on:
  workflow_dispatch:
    inputs:
      target_branch:
        description: 'Deploy from which branch?'
        required: true
        default: 'main'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      with:
        ref: ${{ github.event.inputs.target_branch }}

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Deploy to Azure Web App
      uses: azure/webapps-deploy@v2
      with:
        app-name: 'wadayym'
        slot-name: production
        publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
        package: .
```
4. Azure の発行プロファイルを GitHub に登録
- Azure Portal → App Service → 「発行プロファイルを取得」
→ GitHub のリポジトリに AZURE_WEBAPP_PUBLISH_PROFILE という名前で Secrets に登録
5. デプロイ確認
- GitHub Actions のログで成功を確認し、
https://wadayym.azurewebsites.net にアクセスして動作確認

